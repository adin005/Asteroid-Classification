import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FEATURES } from './feature-schema';
import { PredictionService, PredictionResponse } from './prediction.service';

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly fields = FEATURES;
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);
  values: Record<string, number | string | null> = {};
  loading = signal(false);
  error = signal('');
  result = signal<PredictionResponse | null>(null);
  private api = inject(PredictionService);

  constructor() {
    for (const field of this.fields) this.values[field.name] = null;
  }

  submit() {
    this.error.set('');
    this.result.set(null);
        if (this.deployedStatic) {
      this.error.set('Static GitHub Pages demo: run Flask and Angular locally to get predictions.');
      return;
    }

    const payload: Record<string, number | null> = {};
    for (const field of this.fields) {
      const value = this.values[field.name];
      if (value === null || value === '') {
        if (field.required) {
          this.error.set(`Enter a value for ${field.label}`);
          return;
        }
        payload[field.name] = null;
        continue;
      }
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        this.error.set(`Invalid number for ${field.label}`);
        return;
      }
      if ((field.min !== undefined && numeric < field.min) ||
          (field.max !== undefined && numeric > field.max)) {
        this.error.set(`Out-of-range value for ${field.label}`);
        return;
      }
      payload[field.name] = numeric;
    }

    this.loading.set(true);
    this.api.predict(payload).subscribe({
      next: (res) => { this.result.set(res); this.loading.set(false); },
      error: (err) => {
        this.error.set(`API error ${err.status || ''}: ` +
          (err.error?.error || err.message || 'check the Flask terminal'));
        this.loading.set(false);
      },
    });
  }
}