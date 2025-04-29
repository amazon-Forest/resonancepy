import numpy as np
import time
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import h5py
from scipy.signal import detrend

from resonance_py.drivers.keysightN5221B import KeysightN5221B
from resonance_py.utils.segmentation import create_resonator_segments
from resonance_py.utils.statistics import comp2magdB
from resonance_py.utils.peak_analysis import findPeaks, peak_info
# from resonance_py.drivers.SetAttenuation import set_attenuation
from resonance_py.data_analysis.resonator_data import ResonatorData
from resonance_py.data_analysis.modeling import single_resonator_fit
from resonance_py.utils.logging_utils import (
    setup_logger, get_timestamped_log_filename, 
    log_exception, log_measurement_end
)

from qcodes.dataset.experiment_container import new_experiment
from qcodes import Parameter
from qcodes.dataset import (
    Measurement,
)

class ResonatorMeasurement:
    """
    Class to manage resonator measurements similar to MATLAB's ResonatorMeasurements.
    
    This class handles:
    1. Survey scan to find resonators
    2. Refinement of resonator frequencies
    3. Segmented measurements at different attenuation values
    4. Data analysis and saving results
    """
    
    def __init__(self ,pna:KeysightN5221B | None = None
                 , attenuator=None
                 , settings=None
                 , atten_address=None
                 ):
        """
        Initialize the resonator measurement.
        
        Parameters:
        -----------
        pna : KeysightN5221B or compatible
            VNA instrument object
        save_path : str
            Path to save data
        settings : dict
            Measurement settings
        """
        self.pna = pna
        self.attenuator = attenuator
        self.attenuation = attenuator.attenuation()
        
        self.run_id = {
            "success": 0,
            "fail": 0
        }

        self.survey_runs = {
            "success": 0,
            "fail": 0
        }
        self.refinement_runs = {
            "success": 0,
            "fail": 0
        }
        
        self.segmented_runs = {
            "success": 0,
            "fail": 0
        }
        
        self.full_runs = {
            "success": 0,
            "fail": 0
        }
        

        # Default settings
        self.default_settings = {
            'experiment_name': 'DefaultName',
            'sample_name': 'ResonatorSample',
            'fridge_name': None,
            'num_resonators': 3,
            'selected_resonators': [0, 1, 2],  # Indices of resonators to measure
            'attenuation_values': [0, 10, 20, 50, 75],
            'system_attenuation': 0,  # Attenuation present in the channel
            'target_SNR': 10,  # Minimum SNR for good data
            'max_average': 50,  # Maximum number of averages
            'min_num_sweeps': 1,  # Minimum number of sweeps
            
            'save_settings': {
                'save_base_path': None,  # Path to save settings
                'save_base_file_name': None,  # Base name for saving settings
                'save_all': False,  # Save all data or only selected resonators
                'unique_id': None,  # Unique ID for the measurement
            },
            

            'survey': {
                'points': 1000,
                'if_bandwidth': 2000,
                'measurement': 'S21',
                'save_data': False,
            },
            'refine': {
                'initial_span': 500e6,
                'points': 1000,
                'if_bandwidth': 100,
                'measurement': 'S21',
                'span_refinement_value': 0.5,  # Factor to reduce span on each iteration
                'fwhm_index_width_limit': 80,  # Minimum number of points across FWHM
                'save_data': False,
            },
            'segment': {
                'points': [75, 60, 40, 50],
                'segment_factors': [0.5, 1, 2, 3],  # Width factors for segments
                'measurement': 'S21',
                'averaging': 1,
                'save_data': False,
                'analyze': False,
            }
        }
        
        # Update settings with user-provided values
        self.settings = self.default_settings.copy()
        if settings is not None:
            self.settings.update(settings)

        if self.settings['save_settings']['save_all'] is True:
            self.settings['survey']['save_data'] = True
            self.settings['refine']['save_data'] = True
            self.settings['segment']['save_data'] = True
        elif self.settings['save_settings']['save_all'] is False:
            self.settings['survey']['save_data'] = False
            self.settings['refine']['save_data'] = False
            self.settings['segment']['save_data'] = False

                # Set default save paths
        if self.settings['save_settings']['save_base_path'] is None:
            self.save_path = Path('./data')
        else:
            self.save_path = Path(self.settings['save_settings']['save_base_path'])
        
        if self.settings['save_settings']['save_base_file_name'] is None:
            self.save_base_name = f'{self.settings['experiment_name']}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        else:
            self.save_base_name = self.settings['save_settings']['save_base_file_name']
        
        # Create save directory if it doesn't exist
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Create data directory for plots
        # self.figure_save_path = self.save_path / f'DataGraphs-{datetime.now().strftime("%Y%m%d")}'
        # self.figure_save_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()

        # Measurement results
        self.survey_data = None
        self.refinement_data = None
        self.segment_data = None
        self.full_data = None
        self.resonators = []
        
        # Log initialization
        self.logger.info(f"ResonatorMeasurement initialized with base name: {self.save_base_name}")

        self.logger.info("========== MEASUREMENT STARTED ==========")
        self.logger.info(f"Survey center set to: {settings['survey'].get('centerFrequency'):.5e} Hz")
        self.logger.info(f"Survey span set to: {settings['survey'].get('frequencySpan'):.5e} Hz")
        self.logger.info(f"System Attenuation: {settings.get('system_attenuation')} dB")
        self.logger.info(f"Expected Resonators: {settings.get('num_resonators')}")
        self.logger.info("========================================")
    
    def _setup_logging(self):
        """Setup logging to file and console with enhanced configuration."""
        # Create timestamped log filename
        log_file = get_timestamped_log_filename(
            self.save_path, 
            f"{self.save_base_name}", 
            extension=".log"
        )
        
        # Configure formatter with more detail
        formatter_str = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        # Setup logger using utility function
        self.logger = setup_logger(
            name=f'resonator_measurement_{id(self)}',
            log_file=log_file,
            level=logging.INFO,
            formatter_str=formatter_str,
            console_output=True
        )
        
        # Log the location of the log file
        self.logger.info(f"Logging to file: {log_file}")
        
        # Add an uncaught exception handler to log any unhandled exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupt to avoid clutter
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            self.logger.critical("Uncaught exception", 
                               exc_info=(exc_type, exc_value, exc_traceback))
        
        sys.excepthook = handle_exception
    
    def survey_scan(self, plot=True , save_data=False):
        """
        Perform a wide sweep to find resonators.
        """
        self.logger.info("Starting survey scan...")
        start_time = time.time()
        survey_settings = self.settings['survey']
        
        try:
            # Extract survey settings
            center = survey_settings['centerFrequency']
            span = survey_settings['frequencySpan']
            points = survey_settings['points']
            if_bandwidth = survey_settings['if_bandwidth']
            measurement = survey_settings['measurement']
            is_dip = self.settings.get('is_dip', True)
            fix_zero_widths = self.settings.get('fix_zero_widths', True)


            # Log detailed measurement parameters
            self.logger.info(f"Survey parameters: center={center} Hz, span={span} Hz, points={points}, IF BW={if_bandwidth} Hz")
            
            # Perform the sweep
            complex_data, frequencies = self.pna.linear_sweep(
                center_frequency=center, 
                frequency_span=span, 
                points=points, 
                if_bandwidth=if_bandwidth,
                measurement=measurement
            )
            
            # Convert to magnitude in dB for peak finding
            mag_db = comp2magdB(complex_data)
            
            # Find resonator peaks
            self.logger.info(f"Finding resonator peaks...")
            peaks_info = peak_info(
                data=mag_db, 
                frequencies=frequencies, 
                expectedPeaks=self.settings['num_resonators'],
                is_dip=is_dip,
                fix_zero_peaks=fix_zero_widths
            )
            
            # Log found peaks information
            if 'fo' in peaks_info and len(peaks_info['fo']) > 0:
                self.logger.info(f"Found {len(peaks_info['fo'])} resonator peaks:")
                for i, freq in enumerate(peaks_info['fo']):
                    self.logger.info(f"  Peak {i+1}: {freq/1e9:.6f} GHz, FWHM: {peaks_info['fwhm_freq'][i]/1e6:.3f} MHz")
            else:
                self.logger.warning("No resonator peaks found!")
            
            survey_data = {
                'run_type': 'survey',
                'run_id': sum(self.run_id.values()),
                's21': complex_data,
                'frequencies': frequencies,
                'peaks_info': peaks_info,
                'sweep_settings': {
                    'center': center,
                    'span': span,
                    'points': points,
                    'if_bandwidth': if_bandwidth,
                    'measurement': measurement
                }
            }

            self.survey_data = survey_data
            
            if plot:
                # Create a figure with 2 vertically stacked subplots that share the x-axis
                fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                                            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})

                # Top plot with magnitude and phase
                color1 = 'tab:blue'
                ax1.set_ylabel('Magnitude (dB)', color=color1)
                line1 = ax1.plot(frequencies, mag_db, label='S21 (dB)', color=color1)
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.set_title('Survey Scan')

                # Add phase to the first subplot (twin y-axis)
                ax2 = ax1.twinx()
                color2 = 'tab:green'
                ax2.set_ylabel('Phase (degrees)', color=color2)
                phase = np.angle(complex_data, deg=True)
                phase_unwrapped = np.unwrap(phase, discont=180)
                phase_detrend = detrend(phase_unwrapped)
                diff_phase = np.abs(np.gradient(phase_detrend))

                line2 = ax2.plot(frequencies, phase_detrend, label='Phase (deg)', color=color2, linestyle='--')
                ax2.tick_params(axis='y', labelcolor=color2)

                # Combined legend for top plot
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='best')

                # Bottom plot with just the phase difference
                color3 = 'tab:red'
                ax3.plot(frequencies, diff_phase, label='Phase Difference', color=color3)
                ax3.set_xlabel('Frequency (Hz)')
                ax3.set_ylabel('Phase Difference')
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.legend()

                for index in peaks_info['peakIndex']:
                    ax1.scatter(frequencies[index], mag_db[index], color='red')
                    ax2.scatter(frequencies[index], phase_detrend[index], color='red')
                    ax3.scatter(frequencies[index], diff_phase[index], color='blue')

                # Only show x-axis labels on the bottom plot
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.show()
            
            # Log completion
            end_time = time.time()
            self.logger.info(f"Survey scan completed in {end_time - start_time:.2f} seconds")
            self.survey_runs["success"] += 1
            self.run_id["success"] += 1
            if 'save_data' in self.settings['survey'].keys() and self.settings['survey']['save_data'] is True:
                self._save_data(survey_data)
        
        except Exception as e:
            self.logger.error(f"Error in survey scan: {str(e)}")
            log_exception(self.logger)
            self.survey_runs["fail"] += 1
            self.run_id["fail"] += 1
            raise
    
    def refinement_scan(self, resonator_index, plot=True):
        """
        Refine the measurement for a single resonator.
        
        Parameters:
        -----------
        resonator_index : int
            Index of the resonator in selectedResonators
        peak_index : int
            Index of the peak in the survey data
        
        Returns:
        --------
        dict
            Refined resonator data
        """
        self.logger.info(f"Refining resonator {resonator_index+1}...")
        start_time = time.time()
        is_dip = self.settings.get('is_dip', True)
        fix_zero_widths = self.settings.get('fix_zero_widths', True)
        
        try:
            # Extract settings
            refine_settings = self.settings['refine']
            
            # Get initial resonator frequency
            if self.survey_data is None:
                raise ValueError("Survey scan must be run before refining resonators")
            
            if self.refinement_data is None:
                self.refinement_data = {
                    'resonator': {}  # Dictionary with resonator_index as keys
                }
            # Check if this resonator has been refined before
            if resonator_index not in self.refinement_data['resonator']:
                # Initialize data for this resonator
                self.refinement_data['resonator'][resonator_index] = {
                    'run_data': [],
                }
            current_run = sum(self.refinement_runs.values())
            
            frequencies = self.survey_data['frequencies']
            refined_center = self.survey_data['peaks_info']['fo'][resonator_index]
            refined_span = refine_settings['initial_span']
            refined_if_bandwidth = refine_settings['if_bandwidth']
            refined_points = refine_settings['points']
            
            # Initial FWHM index width
            fwhm_freq = self.survey_data['peaks_info']['fwhm_freq'][resonator_index]
            fwhm_index_width = fwhm_freq / (self.settings['survey']['frequencySpan'] / self.settings['survey']['points'])
            
            refined_loop = 0
            fwhm_index_width_limit = refine_settings['fwhm_index_width_limit']
            span_refinement_value = refine_settings['span_refinement_value']
            
            return_refined = []

            # If initial FWHM is already sufficient, perform at least one refinement scan
            if fwhm_index_width_limit <= fwhm_index_width:
                self.logger.info(f"Initial FWHM index width ({fwhm_index_width:.2f}) is already sufficient (limit: {fwhm_index_width_limit})")
                self.logger.info(f"Performing single refinement scan for resonator {resonator_index+1}...")
                
                survey_settings = self.settings['survey']
                
                center = survey_settings['centerFrequency']
                span = survey_settings['frequencySpan']
                points = self.settings['survey']['points']
                if_bandwidth = survey_settings['if_bandwidth']
                measurement = survey_settings['measurement']
                is_dip = self.settings.get('is_dip', True)
                fix_zero_widths = self.settings.get('fix_zero_widths', True)
                
                # Perform at least one refined sweep
                res_complex_data, res_frequencies = self.pna.linear_sweep(
                center_frequency=center, 
                frequency_span=span, 
                points=points, 
                if_bandwidth=if_bandwidth,
                measurement=measurement
            )
                
                # Convert to magnitude in dB for peak finding
                res_mag_db = comp2magdB(res_complex_data)
                
                # Find resonator peak information
                res_peak_info = peak_info(
                    data=res_mag_db, 
                    frequencies=res_frequencies, 
                    expectedPeaks=1,
                    is_dip=is_dip,
                    fix_zero_peaks=fix_zero_widths
                )
                
                iteration_data = {
                    'resonator': resonator_index,
                    'run_type': 'refinement',
                    'total_run_id': sum(self.run_id.values()),
                    'current_run_id': current_run,
                    'run_id': f'{current_run}_0',
                    'loop_id': refined_loop,
                    's21': res_complex_data,
                    'frequencies': res_frequencies,
                    'peak_info': res_peak_info,
                    'sweep_settings': {
                        'center': refined_center,
                        'span': refined_span,
                        'points': refined_points,
                        'if_bandwidth': refined_if_bandwidth,
                        'measurement': refine_settings['measurement'],
                    }
                }
                
                if 'save_data' in refine_settings.keys() and refine_settings['save_data'] is True:
                    self._save_data(iteration_data)
                
                return_refined.append(iteration_data)
                self.refinement_data['resonator'][resonator_index]['run_data'].append(iteration_data)
                self.refinement_runs['success'] += 1
                
                if plot:
                    plt.figure(figsize=(5, 2))
                    plt.plot(res_frequencies, res_mag_db, label='S21 (dB)')
                    plt.title(f'Resonator #{resonator_index}, Refinement Scan #{refined_loop+1}')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')
                    for index in res_peak_info['peakIndex']:
                        plt.scatter(res_frequencies[index], res_mag_db[index], color='red')
                    plt.show()
            # Refine the measurement until we have enough points across the resonance
            while fwhm_index_width_limit > fwhm_index_width and refined_loop < 10:
                self.logger.info(f"     Resonator {resonator_index+1} refined linear sweep #{refined_loop+1}...")
                self.logger.info(f"     Center: {refined_center/1e9:.6f} GHz, Span: {refined_span/1e6:.3f} MHz")
                
                # Perform refined sweep
                res_complex_data, res_frequencies = self.pna.linear_sweep(
                    center_frequency=refined_center, 
                    frequency_span=refined_span, 
                    points=refined_points, 
                    if_bandwidth=refined_if_bandwidth,
                    measurement=refine_settings['measurement']
                )
                
                # Convert to magnitude in dB for peak finding
                res_mag_db = comp2magdB(res_complex_data)
                
                # Find resonator peak information
                self.logger.info("      Finding resonator peak in refined data...")
                res_peak_info = peak_info(
                    data=res_mag_db, 
                    frequencies=res_frequencies, 
                    expectedPeaks=1,
                    is_dip=is_dip,
                    fix_zero_peaks=fix_zero_widths
                )
                
                # Calculate FWHM index width
                fwhm_index_width = res_peak_info['fwhm_freq'][0] / (refined_span / refined_points)
                self.logger.info(f"     FWHM index width: {fwhm_index_width:.2f}")

                iteration_data = {
                    'resonator': resonator_index,
                    'run_type': 'refinement',
                    "total_run_id"  : sum(self.run_id.values()),
                    'current_run_id': current_run,
                    'run_id': f'{current_run}.{refined_loop}',
                    'loop_id': refined_loop,
                    's21': res_complex_data,
                    'frequencies': res_frequencies,
                    'peak_info': res_peak_info,
                    'sweep_settings': {
                        'center': refined_center,
                        'span': refined_span,
                        'points': refined_points,
                        'if_bandwidth': refined_if_bandwidth,
                        'measurement': refine_settings['measurement'],
                    }
                }
                if 'save_data' in refine_settings.keys() and refine_settings['save_data'] is True:
                    self._save_data(iteration_data)
                else:
                    self.logger.info(f"Refinement data not saved as save_data is {refine_settings['save_data']}")
                
                return_refined.append(iteration_data)
                self.refinement_data['resonator'][resonator_index]['run_data'].append(iteration_data)
                if plot:
                    plt.figure(figsize=(5, 2))
                    plt.plot(frequencies, res_mag_db, label='S21 (dB)')
                    plt.title(f'Resonator #{resonator_index}, Refinement Scan #{refined_loop+1}')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')
                    for index in res_peak_info['peakIndex']:
                        plt.scatter(frequencies[index], res_mag_db[index], color='red')
                    plt.show()

                # Update center and span for next iteration
                refined_center = res_peak_info['fo'][0]
                refined_span = refined_span * span_refinement_value
                refined_loop += 1
                
            refined_data = self.refinement_data['resonator'][resonator_index]['run_data'][-1]
            end_time = time.time()
            self.logger.info(f"Refinement for resonator {resonator_index+1} completed in {end_time - start_time:.2f} seconds")
            self.refinement_runs['success'] += 1
            


            return refined_data, return_refined
        
        except Exception as e:
            self.logger.error(f"Error refining resonator {resonator_index+1}: {str(e)}")
            log_exception(self.logger)
            self.refinement_runs['fail'] += 1
            raise
    
    def segmented_scan(self, resonator_data, plot=True, set_atten=True):
        """
        Perform segmented measurements at different attenuation values.
        
        Parameters:
        -----------
        resonator_index : int
            Index of the resonator
        resonator_data : dict
            Refined resonator data
        
        Returns:
        --------
        dict
            Segmented measurement data
        """  

        analyze = self.settings['segment'].get('analyze', False)
        resonator_index = resonator_data['resonator']
        self.logger.info(f"Starting segmented measurements for resonator {resonator_index+1}")

        if self.segment_data is None:
            self.segment_data = {
                'resonator': {}  # Dictionary with resonator_index as keys
            }
        # Check if this resonator exists in the data
        if resonator_index not in self.segment_data['resonator']:
            # Initialize data for this resonator
            self.segment_data['resonator'][resonator_index] = {
                'run_data': [],
            }
        current_run = sum(self.segmented_runs.values())

        
        # Extract settings
        segment_settings = self.settings['segment']
        attenuation_values = self.settings['attenuation_values']
        atten_address = getattr(self, 'atten_address', None)
        
        # Extract resonator information
        resonator_frequencies = resonator_data['peak_info']['fo']
        fwhm_values = resonator_data['peak_info']['fwhm_freq']
        
        # Create segments for detailed characterization
        self.logger.info("Creating optimized segments...")
        segments = create_resonator_segments(
            resonator_frequencies=resonator_frequencies,
            fwhm_values=fwhm_values,
            f_sec=segment_settings['segment_factors'],
            n_points=segment_settings['points'],
            fill_gaps=False
        )
        return_segments = []
        # Perform segmented sweep for each attenuation value
        for i, atten in enumerate(attenuation_values):
            current_run = sum(self.segmented_runs.values())
            if set_atten:
                self.attenuator.attenuation(atten)
                # self.attenuation = set_attenuation(target_attenuation_value=atten, address=atten_address)
            
            self.logger.info(f"     Performing sweep: resonator {resonator_index+1}, attenuation = {atten} dB, ({i+1}/{len(attenuation_values)})")
            seg_complex_data, seg_frequencies = self.pna.segmented_sweep(
                segments_data=segments,
                averaging=segment_settings['averaging'],
                measurement=segment_settings['measurement']
            )
            
            # Save this segment data
            segment_data = {
                'run_type': 'segmented',
                "total_run_id"  : sum(self.run_id.values()),
                'run_id': f'{current_run}_{i}_{atten}dB',
                'resonator': resonator_index,
                'segmented_run_id': current_run,
                'loop_id': i,
                's21': seg_complex_data,
                'frequencies': seg_frequencies,
                'sweep_settings': {
                        'segments': segments,
                        'attenuation': atten,
                        'points': segment_settings['points'],
                        'segment_factors': segment_settings['segment_factors'],  # Width factors for segments
                        'measurement': segment_settings['measurement'],
                        'averaging': segment_settings['averaging']
                    }
            }
            

            return_segments.append(segment_data)
            self.segment_data['resonator'][resonator_index]['run_data'].append(segment_data)
            if plot:
                plt.figure(figsize=(5, 2))
                plt.plot(seg_frequencies/1e9, comp2magdB(seg_complex_data), label='S21 (dB)')
                plt.title(f'Resonator #{resonator_index}, Segmented Sweep for {atten} dB')
                plt.xlabel('Frequency (GHz)')
                plt.ylabel('Magnitude (dB)')
                plt.show()
            if 'save_data' in segment_settings.keys() and segment_settings['save_data'] is True:
                self._save_data(segment_data)
            
            if analyze:
                self.logger.info(f"Analyzing segmented data for resonator {resonator_index+1}, attenuation = {atten}dB")
                save_to = self.save_path / 'plots'
                plot_name = f"R0{resonator_index+1}_atten{atten}dB"


                save_to_info ={
                    "save_plots": True,  # Save plots
                    "file_type": "png",  # Default file type for saving results
                    "plot_name_format": 'manual',  # Format for saving plots
                    "plot_group": "plots",
                    "show_plots": True,  # Show plots
                    'file_path': save_to,  # Path to save plots
                    'plot_name': plot_name,
                }

                resonator_data = ResonatorData(freq=seg_frequencies,raw_s21=seg_complex_data)
                resonator_data.save_to_info = save_to_info

                resonator_data.atten = atten
                resonator_data.fit.update({'model': 'Probst'})

                resonator_data = single_resonator_fit(resonator_data, opts={"plot":False})
                self.logger.info(f"=============================Finished analyzing segmented data for resonator {resonator_index+1}, data saved to {save_to}==============================")
                time.sleep(2)
            self.segmented_runs['success'] += 1
            self.run_id['success'] += 1
        
        self.logger.info(f"Completed segmented measurements for resonator {resonator_index+1}")
        return return_segments
    
    def full_scan(self, save_data=False):
        """
        Run the complete measurement process.
        
        1. Survey scan to find resonators
        2. Refine each resonator
        3. Perform segmented measurements at different attenuation values
        """
        self.logger.info("Starting complete resonator measurement workflow")
        start_time = time.time()
        
        try:
            # Perform survey scan if not already done
            if self.survey_data is None:
                self.survey_scan(save_data=save_data)
                
            peaks_info = self.survey_data['peaks_info']
            # Process selected resonators
            selected_resonators = self.settings['selected_resonators']
            if selected_resonators is None:
                # Use all resonators if none specified
                selected_resonators = list(range(min(len(peaks_info['peakIndex']), 
                                                     self.settings['num_resonators'])))
            
            self.logger.info(f"Selected resonators: {selected_resonators}")

            if self.full_data is None:
                self.full_data = {
                    'resonator': {}  # Dictionary with resonator_index as keys
                }
            # Check if this resonator exists in the data
            current_run = sum(self.full_runs.values())
            
            # Process each resonator
            for i, res_idx in enumerate(selected_resonators):
                res_start_time = time.time()
                
                if res_idx not in self.full_data['resonator']:
                    # Initialize data for this resonator
                    self.full_data['resonator'][res_idx] = {
                        'run_data': [],
                    }
                
                # Skip if resonator index is out of range
                if res_idx >= len(peaks_info['peakIndex']):
                    self.logger.warning(f"Resonator index {res_idx} out of range, skipping")
                    continue
                
                # Get peak index from survey scan
                peak_idx = peaks_info['peakIndex'][res_idx]
                
                # Refine resonator measurement
                refined_data, all_refined = self.refinement_scan(i)
                
                # Perform segmented measurements at different attenuation values
                segment_data = self.segmented_scan(refined_data)
                
                # Store resonator data
                resonator_data = {
                    'resonator_index': res_idx,
                    'run_id': current_run,
                    'peak_index': peak_idx,
                    'refinement_data': all_refined,
                    'segment_data': segment_data
                }

                self.full_data['resonator'][res_idx]['run_data'].append(resonator_data)
                
                res_end_time = time.time()
                self.logger.info(f"Resonator {i+1} measurement completed in {res_end_time - res_start_time:.2f} seconds")
            
            
            end_time = time.time()
            duration = end_time - start_time
            log_measurement_end(self.logger, duration)

        
        except Exception as e:
            self.logger.error(f"Error during measurement run: {str(e)}")
            log_exception(self.logger)            
            raise
    
    def _save_data(self, data):
        """
        Save data to QCoDeS database.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to save (can be complex or real)
        frequencies : np.ndarray
            Frequency array
        isComplex : bool, default=True
            True if data is complex
        name_suffix : str, default=""
            Optional suffix for experiment name
        
        Returns:
        --------
        int
            Run ID of the saved data
        """

        save_path = Path(self.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        base_name = self.settings['sample_name']

        file_name = self.save_base_name
        unique_id = self.settings['save_settings']['unique_id']
        if type(unique_id) == str:
            file_name = f"{file_name}_{unique_id}"
        elif unique_id == True:
            file_name = f"{file_name}_{id(self)}"
        else:
            file_name = f"{file_name}"
        filepath = save_path / f"{file_name}.h5"
        import json
        
        with h5py.File(filepath, 'a') as f:
            
            run_type = data['run_type']
            run_id = data['run_id']
            if run_type != 'survey':
                resonator_num = str(data['resonator'])
                resonator_name = f'R0{resonator_num}'
                dataset_name = f'{base_name}_{resonator_name}_{run_type[:3]}{run_id}'
            else:
                resonator_name = 'survey'
                dataset_name = f'{base_name}_{run_type}_{run_id}'
            
            if run_type not in f.keys():
                f.create_group(run_type)
            run_type_group = f[run_type]
            
            # create the resonator group if it doesn't exist
            group = f[run_type]
            # create the resonator dataset
            dt = np.dtype([('frequency', np.float64), 
                        ('s21', np.complex128)])
            freq = data['frequencies']
            s21 = data['s21']
            combined_data = np.zeros(len(freq), dtype=dt)
            combined_data['frequency'] = freq
            combined_data['s21'] = s21
            dataset = group.create_dataset(dataset_name, data=combined_data)
            for key, value in data.items():
                if key == 's21' or key == 'frequencies':
                    continue
                elif type(value) == dict:
                    value = self._convert_numpy_to_python(value)
                    value = json.dumps(value)
                dataset.attrs[key] = value
        self.logger.info(f"{data['run_type']} Data saved to {filepath}")


    def _convert_numpy_to_python(self, obj):
        """
        Recursively convert numpy arrays and scalars to Python types.
        
        Parameters:
        -----------
        obj : any
            The object to process
        
        Returns:
        --------
        any
            The object with numpy arrays/scalars converted to Python types
        """
        if isinstance(obj, dict):
            # Process each key-value pair in the dictionary
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
        
        elif isinstance(obj, list) or isinstance(obj, tuple):
            # Process each element in lists or tuples
            return [self._convert_numpy_to_python(item) for item in obj]
        
        elif isinstance(obj, np.ndarray):
            # Convert numpy array to Python list
            return obj.tolist()
        
        elif isinstance(obj, np.number):
            # Convert numpy scalar to Python number
            return obj.item()
        
        elif isinstance(obj, np.bool_):
            # Handle numpy booleans
            return bool(obj)
        
        else:
            # Keep other types unchanged
            return obj