void DoTimeFitForBetaShort(int whichFit, double fitTimeStart) {
    if(whichFit == 0) cout << "******Beta countrate with chi-squared******" << endl;
    if(whichFit == 1) cout << "******Beta countrate with maximum likelihood******" << endl;
    cout << endl;
    
    gStyle->SetHistLineWidth(2);
    gROOT->ForceStyle(); 
    
    TFile *SignalFile = new TFile("histosSignal.root", "READ");
    TFile *BackgroundFile = new TFile("histosBack.root", "READ");
    
    TH1D *hSignal = (TH1D*)SignalFile->Get("h_en_0");
    TH1D *hBackground = (TH1D*)BackgroundFile->Get("h_en_0");
    
    TCanvas *c1 = new TCanvas("c1", "Canvas1", 1920, 1080);
    
    hSignal->Draw();
    hBackground->SetLineColor(kRed);
    hBackground->Draw("same");
    
    int timeStamps = 200;
    double maxTime = 200*3.;
    int BackTimeStamps = 200;
    
    double timeInt = 3;
    
    TH1D *hCountRate = new TH1D("hCountRate", "Countrate in a range; Time /s; Countrate 1/s", timeStamps, 0, maxTime);
    TH1D *hCountRateB = new TH1D("hCountRateB", "", timeStamps, 0, maxTime);
    
    for(int i=0; i<timeStamps; i++) {
        TH1D *hS = (TH1D*)SignalFile->Get(("h_en_0_"+std::to_string(i)).c_str());
        
        TH1D *hB;
        if(i<BackTimeStamps) hB = (TH1D*)BackgroundFile->Get(("h_en_0_"+std::to_string(i)).c_str());
        
        double TCR, eTCR, BCR, eBCR;
        
        if(whichFit == 0) { //chi-squared
            TCR = hS->Integral(0, 1000)/timeInt;
            eTCR = sqrt(hS->Integral(0, 1000))/timeInt;
        
            if(i<BackTimeStamps) BCR = hB->Integral(0, 1000)/timeInt;
            if(i<BackTimeStamps) eBCR = sqrt(hB->Integral(0, 1000))/timeInt;
        } if(whichFit == 1) { //max likelihood
            TCR = hS->Integral(0, 1000);
            eTCR = sqrt(hS->Integral(0, 1000));
        
            if(i<BackTimeStamps) BCR = hB->Integral(0, 1000);
            if(i<BackTimeStamps) eBCR = sqrt(hB->Integral(0, 1000));
        }
        
        hCountRate->SetBinContent(i+1, TCR);
        hCountRate->SetBinError(i+1, eTCR);
        
        if(i<BackTimeStamps) hCountRateB->SetBinContent(i+1, BCR);
        if(i<BackTimeStamps) hCountRateB->SetBinError(i+1, eBCR);
    }
    
    TCanvas *cCR = new TCanvas("cCR", "cCR", 1920, 1080);
    cCR->cd();
    //~ hCountRate->Add(hCountRateB, -1);
    hCountRate->Draw();
    hCountRateB->SetLineColor(kRed);
    hCountRateB->Draw("same");
    
    SaveToTxt("AlBeta.txt", hCountRate);
    SaveToTxt("AlBetaBack.txt", hCountRateB);
    
    std::vector<std::string> isotopes = {"Al28"};
    std::vector<double> decayConstants = {-(0.00514586)};

    TF1 *background = new TF1("background", "[0]", 0, maxTime);
    hCountRateB->Fit(background, "QRSN");

    std::string fitFormula = "";
    for (size_t i = 0; i < isotopes.size(); ++i) {
        fitFormula += "[" + std::to_string(2 * i) + "]*exp([" + std::to_string(2 * i + 1) + "]*x) + ";
    }
    fitFormula += "[14]";

    TF1 *expFit = new TF1("expFit", fitFormula.c_str(), fitTimeStart, maxTime);

    for (size_t i = 0; i < decayConstants.size(); ++i) {
        expFit->FixParameter(2 * i + 1, decayConstants[i]);
    }
    
    expFit->SetParLimits(2, 0, 1000);
    
    expFit->SetParameter(14, background->GetParameter(0));
    expFit->SetParLimits(14, background->GetParameter(0) - 3 * background->GetParError(0), background->GetParameter(0) + 3 * background->GetParError(0));
    
    TFitResultPtr ExpfitResult;

    if (whichFit == 0) ExpfitResult = hCountRate->Fit(expFit, "QSRNM");
    if (whichFit == 1) ExpfitResult = hCountRate->Fit(expFit, "QSRNML");

    expFit->Draw("same");

    vector<int> colors = {kOrange, kBlack, kGreen, kMagenta, kCyan};
    vector<TF1*> fits;

    for (size_t i = 0; i < isotopes.size(); ++i) {
        TF1 *fit = new TF1((isotopes[i] + "Fit").c_str(), "[0]*exp([1]*x)", 0, maxTime);
        fit->SetParameter(0, expFit->GetParameter(2 * i));
        fit->SetParameter(1, expFit->GetParameter(2 * i + 1));
        fit->SetLineColor(colors[i]);
        fit->Draw("same");
        fits.push_back(fit);
    }

    TF1 *BackFit = new TF1("BackFit", "[0]", 0, maxTime);
    BackFit->SetParameter(0, expFit->GetParameter(14));
    BackFit->SetLineColor(kRed);
    BackFit->Draw("same");

    if (whichFit == 0 || whichFit == 1) {
        for (size_t i = 0; i < isotopes.size(); ++i) {
            double param = expFit->GetParameter(2 * i);
            double error = expFit->GetParError(2 * i);
            if (whichFit == 1) {
                param /= timeInt;
                error /= timeInt;
            }
            cout << isotopes[i] << " Count Rate at t = 0: (" << param << " +/- " << error << ") 1/s" << endl;
        }
        double backParam = expFit->GetParameter(14);
        double backError = expFit->GetParError(14);
        if (whichFit == 1) {
            backParam /= timeInt;
            backError /= timeInt;
        }
        cout << "Background Count Rate at t = 0: (" << backParam << " +/- " << backError << ") 1/s" << endl;
    }

    auto covMatrix = ExpfitResult->GetCovarianceMatrix();
    vector<TMatrixD> matrices;
    vector<Double_t*> params;
    Double_t *p_total = expFit->GetParameters();

    for (size_t i = 0; i < isotopes.size(); ++i) {
        matrices.push_back(covMatrix.GetSub(2 * i, 2 * i + 1, 2 * i, 2 * i + 1));
        params.push_back(p_total + 2 * i);
    }

    double integrationTime = maxTime;

    for (size_t i = 0; i < isotopes.size(); ++i) {
        double integral = fits[i]->Integral(0, integrationTime);
        double integralError = fits[i]->IntegralError(0, integrationTime, params[i], matrices[i].GetMatrixArray());
        if (whichFit == 1) {
            integral /= timeInt;
            integralError /= timeInt;
        }
        cout << isotopes[i] << " Beta Number: " << integral << " +/- " << integralError << endl;
    }
}
