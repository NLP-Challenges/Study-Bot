# Konzept für den Chatbot "Data"

Team:

- Tobias Buess
- Yvo Keller
- Alexander Shanmugam

## Fragen

- [ ] Scraping (Live Update der Daten in Knowledge Base notwendig)
- [ ] Verschiedene Embedding Modelle austesten und vergleichen sinnvoll?
- [ ] Ist es möglich/sinnvoll den Chatbot auf richtigem Dataset mit vorgefertigen Fragen & Richtigen antworten evaluieren?
- [ ] Soll der Bot Quellen abgeben können?

## Ziel

Ziel dieser Challenge ist die Entwicklung des Chatbots namens "Data". Er soll den Stundenten vom Studiengang Data Science zur Verfügung stehen, und ihnen Fragen rund um den Studiengang beanworten können und für studentische Anliegen zur Verfügung stehen, wobei er vorgegebenen ethischen Leitlinien folgen soll. Data soll auf Systeme des Studiengangs zugreifen und Standardanfragen mit Hilfe einer Wissensbasis beantworten.

Der Bot soll auch Probleme des Benutzers erkennen und darauf moralisch adäquat reagieren, zum Beispiel mit aufmunternden Worten oder mit der Weitergabe an eine Ansprechperson. Er soll zudem zur Motivation der Studierenden beitragen. In diesem Sinne ist der Chatbot als moralische Maschine in der Tradition von GOODBOT und BESTBOT und als empathiesimulierender Softwareroboter in der Art von SPACE THEA umzusetzen.

Beispiele von Nachrichten, die der Bot beantworten können soll:

- Wie ist der Studiengang Data Science aufgebaut?
- Wer unterrichtet das Modul "Grundlagen Machine Learning"?
- Wie sieht die Leistungsnachweis im Modul "Vertiefung der Analysis" aus?
- Lineare Algebra ist nicht mein Ding, was kann ich tun?
- Ich fühle mich aktuell sehr gestresst und überfordert mit dem Studium und privaten Problemen, und überlege, mein Studium abzubrechen.
- (optional) Was ist eine Lineare Regression?

Die Art von Fragen, die Data beantworten können soll, lassen sich wie folgt kategorisieren:

- Fragen zu Studiengang (z.B. Wie ist der Studiengang Data Science aufgebaut?)
- Fragen zu Personen (z.B. Wer unterrichtet das Modul "Grundlagen Machine Learning"?)
- Fragen zu den Inhalten und dem Aufbau von Modulen (z.B. Wie sieht die Leistungsnachweis im Modul "Vertiefung der Analysis" aus?)
- Fragen zu Ressourcen (z.B. Wo finde ich die Wegleitung?)
- (optional) Fragen zu Lernmaterialien (z.B. Was ist eine Lineare Regression?) <- CHECK: Antwort würde standardmässig auch antrainieretes wissen von GPT nutzen, kann das Unterschieden werden?

## Wissensbasis

Die Wissensbasis soll auf divsersen Quellen aufbauen. Dabei ist es wichtig, statisches und dynamisches Wissen zu Unterscheiden. *Statisches Wissen* ist Wissen, welches sich nicht oder nur selten ändert, und mit Hilfe eines Embedding Modells abgebildet werden und dem Bot als Kontext zur Beantwortung von Nachrichten zur Verfügung gestellt werden kann. *Dynamisches Wissen*  kann sich oft ändern, und soll deshalb wenn der Bedarf erkannt wird, Live von entsprechenden APIs abgefragt werden. 

Die Wissensbasis soll folgende Informationen enthalten:

Statisches Wissen:

- Informationen zum Studiengang (Konzept, Wegleitung, Curriculum, Reglement, FHNW Website zum Studiengang etc.)
- Spaces: Modul-Spaces (Porträt, Lernmaterialien, Aufgaben), jedoch NICHT Beiträge und Kalender (geringer Mehrwert diese Informationen verändern sich oft und können veraltet sein)
- (optional) Spaces: Nutzerprofile (Name, Bild, Kontaktinformationen, aktuelles Semester etc.)
- (optional) PDFs aus Lernmaterialien
- (optional) Links zu externen Lernmaterialien

Dynamisches Wissen:

- Stundenplan (Kalender mit Deep Dives und Zeitpunkt Sprechstunde in den Modulen)

## Privatshpäre

Der Datenschutz muss immer gewährleistet werden. Persönliche Daten sollen nicht an Drittanbieter weitergegeben werden. Das erreichen wir mit folgenden Massnahmen:

- Platzhalter verwenden in Kommunikation mit LLM bei Drittanbiter (e.g. GPT-4) für persönlichen Daten (z.B. Name mit Variable {name} ersetzen)

CHECK: Beduetet das, persönliche Daten nicht in der Wissensbasis speichern (z.B. Nutzerprofile)? Embeddete Chuncks können wir eher schlecht von solchen Inhalten bereinigen. Und wenn in Antworten z.B. der Name des Dozenten vorkommen soll, dann müssten wir diesen auch in der Wissensbasis speichern.

## Ethishe Leitlinien

- Bei Problemem des Benutzers moralisch adäquat reagieren (z.B. Stress im Studium, oder Extrembeispiel bei Suizidgedanken) und an entpsrechende Hilfsstellen weitervermitteln
- Der Chatbot soll eine Persönlichkeit haben (motivierend, humorvoll, empathisch, etc.)

## Design

- Avatar Bild
- (optional) Stimme

## Tech Stack

- LLM API (GPT-3.5/4)
- LangChain (Kommunikation mit LLM, Embeddings etc.)
- Streamlit (Chat Interface)
- Embedding Modelle (OpenAI, BERT von Google)

Task-orientierte Dialog Systeme wie Rasa und Dialogflow CX machen für unseren Use Case wenig Sinn, keine spezifischen Tasks ausgeübt werden müssen (z.B. Änderung in einem System). Es geht viel mehr um die Beantwortung von Fragen. Mit LangChain haben wir dazu eine solide Basis, und bleiben flexibel.

## Evaluierung

Ein Chatbot, der falsche Antworten gibt, oder nicht auf die Fragen des Benutzers eingeht, ist nicht hilfreich. Er soll deshalb auf verschiedene Arten evaluiert werden.

### Qualitativ

- Factuality

### Quantitativ

- Metriken wie Coverage, Tiefe, Genauigkeit

## Was wir NICHT erreichen wollen

TODO

## Milestones

TODO

| Was                                                    | Wann           |
|--------------------------------------------------------|----------------|
| Daten sammeln, einlesen, zuschneiden + Konzept/Planung | 16. März 2023  |
| Feature Engineering erstes ML-Modell                   | 06. April 2023 |
| ML-Modelle und refining Feature Engineering            | 04. Mai 2023   |
| DL-Modelle                                             | 25. Mai 2023   |
| App mit Modell                                         | 15. Juni 2023  |
| Abgabe Challenge                                       | 15. Juni 2023  |
| Vergleich der Modelle                                  | 15. Juni 2023  |
| Präsentation                                           | KW 25/26       |


## Architektur und Vorgehen

TOOD
