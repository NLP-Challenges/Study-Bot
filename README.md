# Konzept für den Chatbot "Data"

Team:

- Tobias Buess
- Yvo Keller
- Alexander Shanmugam

## Fragen

- [ ] Scraping (Live Update der Daten in Knowledge Base notwendig)
- [ ] Verschiedene Embedding Modelle austesten und vergleichen sinnvoll?
- [ ] Ist es möglich/sinnvoll den Chatbot auf angefertigtem Dataset mit vorgefertigen Fragen & Richtigen Ja/Nein antworten quantitativ zu evaluieren?
- [ ] Soll der Bot Quellen angeben können?
- [ ] Macht es Sinn, separat zu Testen wie gut die Abfrage der Embeddings ist, und wie gut das LLM daraus anschliessend eine Antwort generiert?

## Ziel

Ziel dieser Challenge ist die Entwicklung des Chatbots namens "Data". Er soll den Stundenten vom Studiengang Data Science zur Verfügung stehen, und ihnen Fragen rund um den Studiengang beanworten können und für studentische Anliegen zur Verfügung stehen, wobei er vorgegebenen ethischen Leitlinien folgen soll. Data soll auf Systeme des Studiengangs zugreifen und Standardanfragen mit Hilfe einer Wissensbasis beantworten. Wir legen unseren Fokus in der Challenge auf eine gut funktionierende, deutsche Version des Chatbots.

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
- (optional) Fragen zu Lernmaterialien (z.B. Was ist eine Lineare Regression?)

## Wissensbasis

Die Wissensbasis soll auf divsersen Quellen aufbauen. Dabei soll der Bot bei Beantwortung der Fragen mitgeliefierten Kontext priorisieren. Wird eine Frage gestellt wie "Was ist eine lineare Regression?", werden also die Lernmaterialien dazu priorisiert, jedoch kann der Bot auch auf das Wissen im LLM zurückgreifen, wenn relevanter Kontext fehlt.

- Informationen zum Studiengang (Konzept, Handbuch, Curriculum, Reglement, Plagiate, FHNW Website zum Studiengang etc.)
- Spaces: Modul-Spaces (Porträt, Lernmaterialien, Aufgaben), jedoch NICHT Beiträge und Kalender (geringer Mehrwert - diese Informationen verändern sich oft und können veraltet sein)
- (optional) Spaces: Nutzerprofile (Name, Bild, Kontaktinformationen, aktuelles Semester etc.)
- (optional) PDFs aus Lernmaterialien
- (optional) Links zu externen Lernmaterialien

### Considerations

Wir müssen berücksichtigen, dass die Inhalte in Deutsch wie auch Englisch geschrieben sind.

## Privatshpäre

Der Datenschutz muss immer gewährleistet werden. Persönliche Daten sollen nicht an Drittanbieter wie OpenAI weitergegeben werden.
Das erreichen wir mit folgenden Massnahmen:

- Wir nutzen ein Named Entity Recognition Modell, um persönliche Daten wie Namen, Organisationen, Orte, und E-Mail Adessen zu erkennen und durch Platzhalter zu ersetzen. (z.B. Name mit Variable {name} ersetzen)
- Das LLM wird über die vorhandenen Variabeln instruiert, und behandelt diese auch für Antworten Platzhalter. (z.B. "Hallo {name}, wie kann ich dir helfen?")
- Anschliessend können wir die Platzhalter in der generierten Antwort wieder durch die erkannten Entitäten ersetzen, und dem User diese anzeigen.

## Ethische Leitlinien

- Bei Problemem des Benutzers moralisch adäquat reagieren (z.B. Stress im Studium, oder Extrembeispiel bei Suizidgedanken) und an entpsrechende Hilfsstellen weitervermitteln
- Der Chatbot soll eine Persönlichkeit haben (motivierend, humorvoll, empathisch, etc.)

## Design

- Avatar Bild
- (optional) Stimme

## Tech Stack

- LLM API (LLAMA2/GPT-3.5/4)
- LangChain (Kommunikation mit LLM, Embeddings etc.)
- Streamlit (Chat Interface)
- Embedding Modelle (OpenAI, BERT von Google)
- ...

Task-orientierte Dialog Systeme wie Rasa und Dialogflow CX machen für unseren Use Case wenig Sinn, keine spezifischen Tasks ausgeübt werden müssen (z.B. Änderung in einem System). Es geht viel mehr um die Beantwortung von Fragen. Mit LangChain haben wir dazu eine solide Basis, und bleiben flexibel.

## Evaluierung

Ein Chatbot, der falsche Antworten gibt, oder nicht auf die Fragen des Benutzers eingeht, ist nicht hilfreich. Er soll deshalb auf verschiedene Arten evaluiert werden.

### Qualitativ

- Factuality

### Quantitativ

Metriken wie:

- Coverage
- Tiefe
- Genauigkeit

## Was wir NICHT erreichen wollen

- Mehrsprachigkeit unterstützen
- Mehrere LLMs in der Evaluation gegenüberstellen

## Milestones

TODO @ALEX Link auf Taskboard
