import React, { useEffect, useState } from 'react';
import { X, Check } from 'lucide-react';
import {
  InquiryFormData,
  validateContactInquiry,
} from '../data/studioData';
import { useTactileSurface } from '../physics/useTactileSurface';

interface ContactModalProps {
  open: boolean;
  onClose: () => void;
}

const INITIAL_FORM: InquiryFormData = {
  name: '',
  email: '',
  discipline: 'Web Design & Art Direction',
  timeline: 'Q2 2026',
  message: '',
};

export const ContactModal: React.FC<ContactModalProps> = ({ open, onClose }) => {
  const [form, setForm] = useState<InquiryFormData>(INITIAL_FORM);
  const [errors, setErrors] = useState<Partial<Record<keyof InquiryFormData, string>>>({});
  const [submitted, setSubmitted] = useState(false);

  const panelSurface = useTactileSurface<HTMLDivElement>({ variant: 'panel', active: open });
  const closeBtn = useTactileSurface<HTMLButtonElement>({ variant: 'button' });
  const doneBtn = useTactileSurface<HTMLButtonElement>({ variant: 'button', active: true });
  const cancelBtn = useTactileSurface<HTMLButtonElement>({ variant: 'button' });
  const submitBtn = useTactileSurface<HTMLButtonElement>({ variant: 'button', active: true });

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const result = validateContactInquiry(form);
    if (!result.valid) {
      setErrors(result.errors);
      return;
    }
    setErrors({});
    setSubmitted(true);
  };

  const handleResetAndClose = () => {
    setForm(INITIAL_FORM);
    setErrors({});
    setSubmitted(false);
    onClose();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="contact-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 sm:p-6"
    >
      <div
        ref={panelSurface.ref}
        {...panelSurface.tactileProps}
        className="hw-panel relative w-full max-w-lg border border-white/15 bg-[#141417] p-6 sm:p-8"
      >
        <div className="flex items-start justify-between gap-4 border-b border-white/[0.08] pb-5">
          <div>
            <p className="text-[11px] tracking-[0.06em] text-[#8E8E93] uppercase">
              New Commission Inquiry
            </p>
            <h2
              id="contact-modal-title"
              className="mt-1.5 text-2xl font-light tracking-[-0.02em] text-[#F4F4F5]"
            >
              Start a conversation
            </h2>
          </div>
          <button
            ref={closeBtn.ref}
            {...closeBtn.tactileProps}
            type="button"
            onClick={handleResetAndClose}
            aria-label="Close inquiry dialog"
            className="hw-control flex h-10 w-10 items-center justify-center rounded-full border border-white/15 text-[#A1A1AA] hover:border-white/50 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        {submitted ? (
          <div className="py-8">
            <div className="flex h-11 w-11 items-center justify-center rounded-full border border-white/25 bg-white/[0.06] text-white">
              <Check className="h-5 w-5" aria-hidden="true" />
            </div>
            <h3 className="mt-5 text-xl font-normal text-[#F4F4F5]">
              Inquiry received
            </h3>
            <p className="mt-2 text-[14px] leading-relaxed text-[#A1A1AA]">
              Thank you, {form.name.trim()}. Our San Diego studio will review your {form.discipline.toLowerCase()} brief for {form.timeline} and reply to {form.email.trim()} within two business days.
            </p>
            <button
              ref={doneBtn.ref}
              {...doneBtn.tactileProps}
              type="button"
              onClick={handleResetAndClose}
              className="hw-control mt-7 rounded-full border border-white/30 bg-[#F4F4F5] px-6 py-2.5 text-[12px] font-medium tracking-[0.06em] text-[#111113] uppercase hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              Done
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} noValidate className="mt-6 space-y-5">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label
                  htmlFor="inquiry-name"
                  className="block text-[12px] font-medium text-[#D4D4D8]"
                >
                  Name
                </label>
                <input
                  id="inquiry-name"
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="Elena Vance"
                  className="mt-1.5 w-full border border-white/15 bg-[#101012] px-3.5 py-2.5 text-[14px] text-[#F4F4F5] placeholder-[#52525B] focus:border-white/60 focus:outline-none"
                />
                {errors.name && (
                  <p role="alert" className="mt-1.5 text-[12px] text-[#FCA5A5]">
                    {errors.name}
                  </p>
                )}
              </div>

              <div>
                <label
                  htmlFor="inquiry-email"
                  className="block text-[12px] font-medium text-[#D4D4D8]"
                >
                  Email
                </label>
                <input
                  id="inquiry-email"
                  type="email"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  placeholder="elena@company.com"
                  className="mt-1.5 w-full border border-white/15 bg-[#101012] px-3.5 py-2.5 text-[14px] text-[#F4F4F5] placeholder-[#52525B] focus:border-white/60 focus:outline-none"
                />
                {errors.email && (
                  <p role="alert" className="mt-1.5 text-[12px] text-[#FCA5A5]">
                    {errors.email}
                  </p>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label
                  htmlFor="inquiry-discipline"
                  className="block text-[12px] font-medium text-[#D4D4D8]"
                >
                  Discipline
                </label>
                <select
                  id="inquiry-discipline"
                  value={form.discipline}
                  onChange={(e) =>
                    setForm({ ...form, discipline: e.target.value })
                  }
                  className="mt-1.5 w-full border border-white/15 bg-[#101012] px-3.5 py-2.5 text-[14px] text-[#F4F4F5] focus:border-white/60 focus:outline-none"
                >
                  <option value="Web Design & Art Direction">
                    Web Design &amp; Art Direction
                  </option>
                  <option value="WebGL2 & Spatial 3D">
                    WebGL2 &amp; Spatial 3D
                  </option>
                  <option value="Design System Architecture">
                    Design System Architecture
                  </option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="inquiry-timeline"
                  className="block text-[12px] font-medium text-[#D4D4D8]"
                >
                  Target Window
                </label>
                <select
                  id="inquiry-timeline"
                  value={form.timeline}
                  onChange={(e) =>
                    setForm({ ...form, timeline: e.target.value })
                  }
                  className="mt-1.5 w-full border border-white/15 bg-[#101012] px-3.5 py-2.5 text-[14px] text-[#F4F4F5] focus:border-white/60 focus:outline-none"
                >
                  <option value="Q2 2026">Q2 2026</option>
                  <option value="Q3 2026">Q3 2026</option>
                  <option value="Q4 2026">Q4 2026</option>
                </select>
              </div>
            </div>

            <div>
              <label
                htmlFor="inquiry-message"
                className="block text-[12px] font-medium text-[#D4D4D8]"
              >
                Project Brief
              </label>
              <textarea
                id="inquiry-message"
                rows={4}
                value={form.message}
                onChange={(e) => setForm({ ...form, message: e.target.value })}
                placeholder="Tell us about the scope, goals, and key launch milestones."
                className="mt-1.5 w-full resize-none border border-white/15 bg-[#101012] px-3.5 py-2.5 text-[14px] text-[#F4F4F5] placeholder-[#52525B] focus:border-white/60 focus:outline-none"
              />
              {errors.message && (
                <p role="alert" className="mt-1.5 text-[12px] text-[#FCA5A5]">
                  {errors.message}
                </p>
              )}
            </div>

            <div className="flex items-center justify-end gap-3 border-t border-white/[0.08] pt-5">
              <button
                ref={cancelBtn.ref}
                {...cancelBtn.tactileProps}
                type="button"
                onClick={handleResetAndClose}
                className="hw-control rounded-full border border-white/20 px-5 py-2.5 text-[12px] font-medium text-[#D4D4D8] hover:border-white/50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
              >
                Cancel
              </button>
              <button
                ref={submitBtn.ref}
                {...submitBtn.tactileProps}
                type="submit"
                className="hw-control rounded-full bg-[#F4F4F5] px-6 py-2.5 text-[12px] font-medium tracking-[0.06em] text-[#111113] uppercase hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
              >
                Send Inquiry
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
