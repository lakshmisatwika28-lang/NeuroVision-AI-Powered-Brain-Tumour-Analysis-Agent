import { BrainCircuit, Dna, Radiation, Users, Clock, Stethoscope } from 'lucide-react';
import Disclaimer from '../components/common/Disclaimer';
import '../styles/page.css';
import './ResearchPage.css';

const TYPES = [
  {
    name: 'Glioma',
    text: 'Tumors arising from glial cells, the supportive tissue of the brain. Gliomas vary widely in growth rate and behavior.',
  },
  {
    name: 'Meningioma',
    text: 'Tumors arising from the meninges, the membranes surrounding the brain and spinal cord. Most are slow-growing.',
  },
  {
    name: 'Pituitary tumor',
    text: 'Tumors of the pituitary gland, which can affect hormone regulation depending on size and type.',
  },
];

const RISK_FACTORS = [
  { icon: Dna, text: 'Certain inherited genetic conditions (e.g. neurofibromatosis, Li-Fraumeni syndrome)' },
  { icon: Radiation, text: 'Previous exposure to ionizing radiation, including some past radiation therapy' },
  { icon: Clock, text: 'Age-related factors — risk profiles differ across age groups' },
  { icon: Users, text: 'Family history, relevant mainly for certain rare hereditary conditions' },
];

const SYMPTOMS = [
  'Persistent or worsening headaches',
  'Seizures, including new-onset seizures',
  'Changes in vision',
  'Weakness or numbness in part of the body',
  'Balance or coordination difficulties',
  'Cognitive or behavioral changes',
];

export default function ResearchPage() {
  return (
    <div className="page">
      <div className="page__inner">
        <div className="page__header">
          <h1>Brain Tumor Research &amp; Prevention</h1>
          <p>Educational background to support understanding of NeuroVision's research context.</p>
        </div>

        <Disclaimer>
          This page provides general educational information only. It is not medical advice and
          should not be used to diagnose, prevent, or treat any condition.
        </Disclaimer>

        <section className="research-section panel">
          <div className="research-section__head">
            <BrainCircuit size={18} />
            <h2>What is a brain tumor?</h2>
          </div>
          <p>
            A brain tumor is an abnormal growth of cells within the brain or its surrounding
            structures. Some grow slowly and stay contained, while others grow more quickly or
            spread into nearby tissue. Not all brain tumors are cancerous — many are benign, and
            behavior varies significantly by type, location, and individual case.
          </p>
        </section>

        <section className="research-section panel">
          <h2>Types covered by NeuroVision</h2>
          <div className="research-types">
            {TYPES.map((t) => (
              <div className="research-type" key={t.name}>
                <h3>{t.name}</h3>
                <p>{t.text}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="research-section panel">
          <h2>Possible risk factors / associations</h2>
          <p className="research-section__note">
            Having one or more of these factors does not mean a tumor will develop, and many
            brain tumors occur with no identifiable or preventable cause. Prevention in the
            traditional sense is often not possible.
          </p>
          <ul className="research-factor-list">
            {RISK_FACTORS.map((f) => (
              <li key={f.text}>
                <f.icon size={15} />
                <span>{f.text}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="research-section panel">
          <h2>Symptoms that are sometimes reported</h2>
          <div className="research-symptoms">
            {SYMPTOMS.map((s) => (
              <span key={s} className="research-symptom">{s}</span>
            ))}
          </div>
        </section>

        <section className="research-section panel research-section--highlight">
          <div className="research-section__head">
            <Stethoscope size={18} />
            <h2>When to seek medical attention</h2>
          </div>
          <p>
            Persistent, worsening, or unusual neurological symptoms — such as those listed above —
            generally warrant evaluation by a qualified healthcare professional. Only a clinician,
            using appropriate diagnostic tools, can assess and interpret symptoms for an
            individual. NeuroVision is a research prototype and is not a substitute for that
            evaluation.
          </p>
        </section>
      </div>
    </div>
  );
}
