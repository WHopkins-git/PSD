"""
Isotope identification from gamma-ray spectra
"""

import numpy as np
from ..utils.isotope_library import ISOTOPE_LIBRARY


def match_peaks_to_library(measured_peaks, library=None, tolerance_keV=5):
    """
    Match measured peaks to isotope library

    Parameters:
    -----------
    measured_peaks : array
        Measured peak energies (keV)
    library : dict, optional
        Isotope library (uses ISOTOPE_LIBRARY if None)
    tolerance_keV : float
        Matching tolerance (keV)

    Returns:
    --------
    matches : list of dict
        Each match contains:
        - measured_energy
        - library_energy
        - isotope
        - nuclide
        - confidence
    """
    if library is None:
        library = ISOTOPE_LIBRARY

    matches = []

    for peak_energy in measured_peaks:
        best_match = None
        best_distance = float('inf')

        for isotope, lines in library.items():
            for line in lines:
                # Handle both formats: float or tuple (energy, nuclide)
                if isinstance(line, (tuple, list)):
                    lib_energy, nuclide = line[0], line[1]
                else:
                    lib_energy, nuclide = line, isotope

                distance = abs(peak_energy - lib_energy)

                if distance < tolerance_keV and distance < best_distance:
                    best_distance = distance
                    best_match = {
                        'measured_energy': peak_energy,
                        'library_energy': lib_energy,
                        'isotope': isotope,
                        'nuclide': nuclide,
                        'distance_keV': distance,
                        'confidence': 1.0 - (distance / tolerance_keV)
                    }

        if best_match is not None:
            matches.append(best_match)

    return matches


def identify_decay_chains(matches):
    """
    Identify decay series from matched peaks

    Parameters:
    -----------
    matches : list
        List of peak matches from match_peaks_to_library()

    Returns:
    --------
    series : dict
        Dictionary with isotopes as keys, containing:
        - confidence_score: overall confidence (0-1)
        - peaks_matched: number of peaks matched
        - peaks_expected: total peaks in library
        - matched_energies: list of matched energies
    """
    if not matches:
        return {}

    # Group matches by isotope
    isotope_matches = {}
    for match in matches:
        isotope = match['isotope']
        if isotope not in isotope_matches:
            isotope_matches[isotope] = []
        isotope_matches[isotope].append(match)

    # Calculate confidence for each isotope
    series = {}
    for isotope, iso_matches in isotope_matches.items():
        # Get expected number of peaks from library
        if isotope in ISOTOPE_LIBRARY:
            expected_peaks = len(ISOTOPE_LIBRARY[isotope])
        else:
            expected_peaks = len(iso_matches)  # Fallback

        matched_peaks = len(iso_matches)

        # Confidence based on:
        # 1. Fraction of expected peaks found
        # 2. Average match confidence
        frac_found = matched_peaks / expected_peaks
        avg_confidence = np.mean([m['confidence'] for m in iso_matches])

        # Combined confidence
        confidence_score = 0.7 * frac_found + 0.3 * avg_confidence

        # Extract matched energies
        matched_energies = [m['measured_energy'] for m in iso_matches]

        series[isotope] = {
            'confidence_score': confidence_score,
            'peaks_matched': matched_peaks,
            'peaks_expected': expected_peaks,
            'matched_energies': matched_energies,
            'match_details': iso_matches
        }

    return series


def identify_isotopes(spectrum_energies, prominence=50, distance=20,
                     tolerance_keV=10, min_confidence=0.5):
    """
    Complete isotope identification workflow

    Parameters:
    -----------
    spectrum_energies : array
        Event energies (keV)
    prominence : float
        Peak finding prominence
    distance : int
        Minimum peak separation (bins)
    tolerance_keV : float
        Peak matching tolerance
    min_confidence : float
        Minimum confidence for reporting

    Returns:
    --------
    results : dict
        Complete identification results
    """
    from .peak_finding import find_peaks_in_spectrum

    # Create histogram
    hist, bins = np.histogram(spectrum_energies, bins=3000, range=(0, 3000))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Find peaks
    peak_energies, peak_counts, _ = find_peaks_in_spectrum(
        bin_centers, hist, prominence=prominence, distance=distance
    )

    # Match to library
    matches = match_peaks_to_library(peak_energies, tolerance_keV=tolerance_keV)

    # Identify series
    series = identify_decay_chains(matches)

    # Filter by confidence
    filtered_series = {iso: info for iso, info in series.items()
                      if info['confidence_score'] >= min_confidence}

    # Sort by confidence
    sorted_series = dict(sorted(filtered_series.items(),
                               key=lambda x: x[1]['confidence_score'],
                               reverse=True))

    results = {
        'peaks_found': peak_energies.tolist(),
        'matches': matches,
        'identified_isotopes': sorted_series,
        'summary': _create_summary(sorted_series)
    }

    return results


def _create_summary(series_dict):
    """Create human-readable summary"""
    if not series_dict:
        return "No isotopes identified above confidence threshold."

    lines = ["Identified Isotopes:"]
    for isotope, info in series_dict.items():
        conf_pct = info['confidence_score'] * 100
        lines.append(f"  • {isotope}: {info['peaks_matched']}/{info['peaks_expected']} peaks, "
                    f"confidence={conf_pct:.1f}%")

    return "\n".join(lines)


def generate_isotope_report(identification_results, sample_info=None):
    """
    Generate comprehensive isotope identification report

    Parameters:
    -----------
    identification_results : dict
        Results from identify_isotopes()
    sample_info : dict, optional
        Sample metadata (name, date, location, etc.)

    Returns:
    --------
    report : str
        Formatted text report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ISOTOPE IDENTIFICATION REPORT")
    lines.append("=" * 60)

    # Sample info
    if sample_info:
        lines.append("\nSample Information:")
        for key, value in sample_info.items():
            lines.append(f"  {key}: {value}")

    # Peak summary
    lines.append(f"\nPeaks Found: {len(identification_results['peaks_found'])}")

    # Matched peaks
    lines.append(f"\nMatched Peaks: {len(identification_results['matches'])}")
    for match in identification_results['matches']:
        lines.append(f"  {match['measured_energy']:.1f} keV → "
                    f"{match['library_energy']:.1f} keV ({match['nuclide']}, "
                    f"confidence={match['confidence']:.3f})")

    # Identified isotopes
    lines.append("\n" + identification_results['summary'])

    # Detailed breakdown
    lines.append("\nDetailed Analysis:")
    for isotope, info in identification_results['identified_isotopes'].items():
        lines.append(f"\n  {isotope}:")
        lines.append(f"    Confidence Score: {info['confidence_score']:.3f}")
        lines.append(f"    Peaks Matched: {info['peaks_matched']}/{info['peaks_expected']}")
        lines.append(f"    Energies: {', '.join([f'{e:.1f}' for e in info['matched_energies']])} keV")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
