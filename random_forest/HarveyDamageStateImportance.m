
X = table(Harvey_Latitude,Harvey_Longitude,Year_Built,Harvey_Distance_From_Shoreline,Harvey_Roof_Type,Harvey_Random_Column,First_Floor_Elevation);


%For the regression trees

t = templateTree('NumVariablesToSample','all',...
    'PredictorSelection','interaction-curvature','Surrogate','off');

%Need 'Surrogate', 'on' for missing data

rng(1);
Mdl = fitrensemble(X,Harvey_Damage_State,'Method','Bag','NumLearningCycles',200, ...
    'Learners',t);

impOOB = oobPermutedPredictorImportance(Mdl);

figure
bar(impOOB)
title('Unbiased Predictor Importance Estimates for Hurricane Harvey Data')
xlabel('Predictor variable')
ylabel('Importance')
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';