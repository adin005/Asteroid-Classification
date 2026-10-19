export type Feature = {
  name: string;
  label: string;
  required: boolean;
  min?: number;
  max?: number;
};

export const FEATURES: Feature[] = [
  { name: 'e',        label: 'Eccentricity (e)',            required: true, min: 0 },
  { name: 'a',        label: 'Semi-major axis (a, AU)',     required: true },
  { name: 'q',        label: 'Perihelion distance (q, AU)', required: true, min: 0 },
  { name: 'i',        label: 'Inclination (i, degrees)',    required: true, min: 0, max: 180 },
  { name: 'tp',       label: 'Perihelion time (tp, JD)',    required: true },
  { name: 'H',        label: 'Absolute magnitude (H)',      required: true },
  { name: 'diameter', label: 'Diameter (km, optional)',     required: false, min: 0 },
  { name: 'albedo',   label: 'Albedo (0 to 1, optional)',   required: false, min: 0, max: 1 },
  { name: 'moid',     label: 'Earth MOID (AU)',             required: true, min: 0 },
];