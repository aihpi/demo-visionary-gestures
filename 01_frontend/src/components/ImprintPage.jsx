import { useNavigate } from 'react-router-dom'

function ImprintPage() {
    const navigate = useNavigate()

    return (
        <div className="h-dvh w-full flex flex-col overflow-hidden">
            <button
                onClick={() => navigate(-1)}
                className="fixed bottom-4 left-4 w-12 h-12 text-xl font-bold border-2 border-main bg-transparent text-main hover:bg-main hover:text-[#fbf4ed] transition-colors duration-250 cursor-pointer flex items-center justify-center z-50 backdrop-blur-lg"
                title="Close"
            >
                ✖
            </button>


            <div className="flex-1 overflow-y-auto px-[4vw] pb-[4vh] pt-[4vh]">
                <div className="max-w-[80vw] mx-auto">
                    <div className="border-2 border-main p-[2vmin]">
                        <h2 className="text-left font-bold text-main mb-[1vh] text-[3vmin] leading-tight">
                            Imprint
                        </h2>
                        <div className="font-medium text-gray-900 text-left text-[1.5vmin] leading-relaxed space-y-4">
                            <div>
                                <strong>Hasso-Plattner-Institut für Digital Engineering gGmbH</strong><br/>
                                Prof.-Dr.-Helmert-Str. 2-3<br/>
                                14482 Potsdam<br/>
                                Phone: +49 (0)331 5509-0<br/>
                                Fax: +49 (0)331 5509-129<br/>
                                Email: hpi-info(at)hpi.de<br/>
                                Website: <a href="https://www.hpi.de" target="_blank" rel="noopener noreferrer" className="text-main underline hover:font-bold">www.hpi.de</a>
                            </div>

                            <div>
                                <strong>Authorized Representative Managing Directors</strong><br/>
                                Prof. Dr. Tobias Friedrich<br/>
                                Dr. Henrik Haenecke
                            </div>

                            <div>
                                <strong>Registry Office</strong><br/>
                                Potsdam District Court<br/>
                                Register Number: HRB 12184 P<br/>
                                Tax ID: DE812987194
                            </div>

                            <div>
                                <strong>Responsible for Content</strong><br/>
                                Prof. Dr. Tobias Friedrich<br/>
                                Dr. Henrik Haenecke
                            </div>

                            <div>
                                <strong>Editor, Web Design, Web Development, Texts, Neural Network</strong><br/>
                                AI Service Centre Berlin Brandenburg
                            </div>

                            <div>
                                <strong>Photos</strong><br/>
                                Unless otherwise stated, the pictures are created during the work of the AI Service Centre Berlin Brandenburg.
                            </div>

                            <div>
                                <strong>Data</strong><br/>
                                All data is created and stored locally in your browser. No data is transmitted elsewhere.
                            </div>

                            <h3 className="font-bold text-main mt-6 mb-2 text-[2.5vmin]">Legal Information</h3>
                            <p>
                                The Hasso Plattner Institute (HPI) constantly checks and updates the information on its web pages. Despite the utmost care taken, we cannot rule out that some of the data may have become outdated. Therefore, we cannot accept liability for the currency, accuracy, and completeness of the information displayed. The same applies to all other web pages that can be reached via hyperlink. HPI is not responsible for the content of websites which can be reached via such links.
                            </p>
                            <p>
                                Furthermore, HPI reserves the right to make changes or additions to the information displayed. Content and structure of the website are protected by copyright. The reproduction of information or data, particularly the use of texts, text excerpts or illustrative material, requires prior approval.
                            </p>

                            <h3 className="font-bold text-main mt-6 mb-2 text-[2.5vmin]">Data Privacy / Privacy Policy</h3>
                            <p className="italic">Last Updated: November 1, 2025</p>

                            <h4 className="font-bold mt-4 mb-1">1. Introduction</h4>
                            <p>
                                We respect your privacy and are committed to protecting your personal data. This privacy policy explains how we collect, use, and protect your information when you use this publication overview page of the AI Service Centre Berlin Brandenburg from Hasso Plattner Institute (the "Service"). This policy complies with the General Data Protection Regulation (GDPR) and other applicable data protection laws.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">2. Data Controller</h4>
                            <p>
                                Hasso-Plattner-Institut für Digital Engineering gGmbH<br/>
                                Prof.-Dr.-Helmert-Str. 2–3<br/>
                                14482 Potsdam, Germany
                            </p>
                            <p>
                                <strong>Data Protection Officer</strong><br/>
                                TÜV SÜD Akademie GmbH<br/>
                                Westendstr. 199<br/>
                                80686 München, Germany<br/>
                                Tel.: +49 (0)381 817082-298<br/>
                                Email: datenschutz(at)hpi.de
                            </p>

                            <h4 className="font-bold mt-4 mb-1">3. Data We Collect</h4>
                            <p>
                                We collect technical data (IP address, browser, device, OS) to ensure functionality, security, and service delivery. This technical data may contain information that identifies you personally.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">4. How We Use Your Data</h4>
                            <p>
                                Collected data is used for service delivery, communication, legal compliance, and protection of our rights. Personal data provided voluntarily will be processed only for stated purposes.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">5. Legal Basis for Processing (GDPR)</h4>
                            <p>
                                We process data based on your consent, contract performance, legitimate interests, or legal obligations (Art. 6 GDPR).
                            </p>

                            <h4 className="font-bold mt-4 mb-1">6. Data Sharing and Disclosure</h4>
                            <p>
                                Your data remains within our responsibility. In special cases (legal disputes, partnerships, service providers), we may share data under strict contractual obligations and GDPR compliance. Transfers outside the EU are communicated explicitly.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">7. Data Storage and Security</h4>
                            <p>
                                We use secure servers, encrypted transmission (HTTPS/TLS), access control, and backup/recovery measures. Data is retained only as necessary or legally required.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">8. Cookies and Tracking</h4>
                            <p>
                                Essential cookies are used for basic functionality (e.g., session management). You can manage cookie preferences in your browser settings. Disabling essential cookies may affect functionality.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">9. Children's Privacy</h4>
                            <p>
                                The Service is not intended for children under 16. We do not knowingly collect personal data from children. If you believe we have collected such data, please contact us immediately.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">10. Your Rights (GDPR)</h4>
                            <p>
                                You have the right to access, correct, delete, restrict, object, and port your data, as well as withdraw consent. To exercise these rights, contact us at ki-servicezentrum(at)hpi.de or via provided unsubscribe links. You also have the right to lodge a complaint with the supervisory authority:
                            </p>
                            <p>
                                Landesbeauftragte für den Datenschutz und für das Recht auf Akteneinsicht<br/>
                                Stahnsdorfer Damm 77<br/>
                                14532 Kleinmachnow<br/>
                                Tel: +49 (0)33203 356-0<br/>
                                Fax: +49 (0)33203 356-49<br/>
                                Email: poststelle(at)lda.brandenburg.de
                            </p>

                            <h4 className="font-bold mt-4 mb-1">11. Automated Decision-Making and Profiling</h4>
                            <p>
                                Your data will not be used for automated decision-making or profiling.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">12. Voluntary Data Provision</h4>
                            <p>
                                Providing personal data is voluntary. Certain procedures (event registration, newsletter) may require specific information, which will be communicated explicitly.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">13. Scientific Data and Research</h4>
                            <p>
                                AI Service Centre Berlin Brandenburg and HPI provide access to published project activities. Custom sequences are stored locally in your browser; you retain ownership.
                            </p>

                            <h4 className="font-bold mt-4 mb-1">14. Right to Object</h4>
                            <p>
                                You may object to processing under Art. 21 GDPR, including direct marketing. Legitimate grounds may override objections only if legally necessary.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ImprintPage