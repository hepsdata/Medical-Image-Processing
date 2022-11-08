ThreeDimTable = zeros(3,4,4); % 3*4 size (3 methods, 4 score, 4 image ; 3*4*4 size matrix)
for index = 1:4 % c001~006 four time repeat
    % load .mat file using for loop & if statement
    if index == 1
        load c001_sa.mat
    end
    if index == 2
        load c002_sa.mat
    end
    if index == 3
        load c003_sa.mat
    end
    if index == 4
        load c006_sa.mat
    end
    %% figure 1 
     % #2-1
        mask_1 = bw1

        figure;
        sgtitle('Figure 1. Overlay The Reference Mask') % add a title to the figure
        subplot(131);
        [rows, columns, numberOfColorChannels] = size(img1); % get the size of the image
        imshow(img1); title('Heart Section') % display the image
        
    
        subplot(132);
        imshow(mask_1); title('Reference Mask') %reference mask 

        subplot(133);
        imshow(img1, []); % display the image
        hold on;
        contour(mask_1, [0.5 0.5], 'r')  % overlay the reference mask with the image
        title('Overlay image')
    
    
    %% figure 2
    % #2-2
    
    reg_max_dist = 30;
    
    figure;
    imshow(img1, []);
    title('Input the Region Growing Seed')
    p = ginput(1);
    % seed 1
    x = round(p(1, 2)); % x?넫?슦紐?
    y = round(p(1, 1)); % y?넫?슦紐?
    
    mask_2_1 = regiongrowing(single(img1), x, y, reg_max_dist);
    mask_2 = activecontour(img1, mask_2_1, reg_max_dist);
    %%
    figure;
    sgtitle('Figure 2. The Region Growing') % add a title to the figure

    subplot(131);
    imshow(img1, []);
    hold on; 
    contour(mask_2, [0.5 0.5], 'y')
    title('Region Growing Segmented Contour') % display the image with the contour(region growing mask)
    
    subplot(132);
    imshow(img1, []);
    hold on;
    contour(mask_1, [0.5 0.5], 'r') % display the image with the contour(reference mask)
    title('Reference Contour')
    
    subplot(133);
    imshow(img1, []);
    hold on;
    contour(mask_2, [0.5 0.5], 'y')
    hold on;
    contour(mask_1, [0.5 0.5], 'r') 
    title('Overlay Contour')
    
    %% figure 3
    % #2-3
    stats = regionprops('table', mask_2, 'Area', 'BoundingBox', 'Centroid', 'Circularity', 'Solidity', 'MajorAxisLength','MinorAxisLength', 'Perimeter', 'ConvexHull');
    for jj=1:size(stats, 1) 
    a = stats.ConvexHull(jj); % convex hull
    b = a{1};
    end
    
    % c = [b(:,1),  b(:,2)]; 

    % mask_3 = poly2mask(b(:,1), b(:,2), 272, 256); % func, which transform polygon to mask

    mask_3 = roipoly(zeros(size(img1)), b(:,1), b(:,2)); % func, which transform polygon to mask
    figure
    subplot(141)
    imshow(img1, []);
    hold on
    title('Region Growing and Convex Hull Segmented Contour')
    contour(mask_3, [0.5 0.5], 'g') % display the image with the contour(region growing + convex hull mask)
    
    subplot(142)
    imshow(img1, []);
    hold on; 
    contour(mask_2, [0.5 0.5], 'y') % display the image with the contour(region growing mask)
    title('Region Growing Segmented Contour')
    
    subplot(143)
    imshow(img1, []);
    hold on;    
    contour(mask_1, [0.5 0.5], 'r')
    title('Reference Contour') % display the image with the contour(reference mask)
    
    subplot(144)
    imshow(img1, [])
    hold on
    contour(mask_3, [0.5 0.5], 'g')
    hold on; 
    contour(mask_2, [0.5 0.5], 'y')
    hold on;
    contour(mask_1, [0.5 0.5], 'r')
    
    title('Overlay Contour')
    
    sgtitle('Figure 3. The Convex Hull Segmentation')
    
    %%
    % % % % scatter & plot the line of the polygon by Convex Hull methods
    % % % % for jj=1:size(stats, 1)
    % % % %     a = stats.ConvexHull(jj);
    % % % %     b = a{1};
    % % % % %     scatter(b(:,1), b(:,2), 5, 'g', 'filled');
    % % % % %     plot(b(:,1), b(:,2), 'g');
    % % % %     hold on;
    % % % % end
    %%
    % % imshow(img1, []);
    % % hold on;
    % % for niter = 100
    % %     if niter>0
    % %         bw_3 = activecontour(img1, bw_3, niter);
    % %     end
    % % end
    % % contour(bw_3, [0.5 0.5], 'g')
    
    %% figure 4
    % #2-4
    
    figure
    imshow(img1)
    roi1 = drawfreehand('color', 'b'); % draw a freehand ROI
    title('Draw the Mask') % draw the mask
    
    mask_4 = createMask(roi1); % create mask
    mask_4 = logical(mask_4); % convert to logical type
    
    for niter = 20
        if niter > 0
            mask_4 = activecontour(img1, mask_4, niter); % active contour method
        end
    end
    %%
    figure
    subplot(141)
    imshow(img1, [])
    hold on;
    contour(mask_4, [0.5,0.5], 'b') % display the image with the contour(level set mask)
    title('Level set segmentation Contour')
    
    subplot(142)
    imshow(img1, [])
    hold on
    contour(mask_3, [0.5 0.5], 'g') % display the image with the contour(region growing + convex hull mask)
    title('Region Growing and Convex Hull Segmented Contour')

    subplot(143)
    imshow(img1, [])
    hold on; 
    contour(mask_2, [0.5 0.5], 'y') % display the image with the contour(region growing mask)
    title('Region Growing Segmented Contour')

    subplot(144)
    imshow(img1, [])
    hold on;
    contour(mask_1, [0.5 0.5], 'r') % display the image with the contour(reference mask)
    title('Reference Contour')
    sgtitle('Figure 4. The Level Set Segmentation')
    
    figure
    imshow(img1, [])
    hold on
    contour(mask_3, [0.5 0.5], 'g')
    hold on; 
    contour(mask_2, [0.5 0.5], 'y')
    hold on;
    contour(mask_1, [0.5 0.5], 'r')
    hold on;
    contour(mask_4, [0.5,0.5], 'b')
    title('Overlay Contour')
    
    %% figure 5
    %dice_ = dice(R, T) / iou = jaccard(R, T)
    % tp = sum((R == 1) & (T == 1)) / fp = sum((R == 0) & (T == 1)) / fn = sum((R == 1) & (T == 0))
    % precis = tp / (tp + fp) / recall = tp / (tp + fn)
    % T : tagert binary image, R : reference binary image

    R = logical(mask_1); % reference mask
    mask_2 = logical(mask_2); % seeded region growing
    mask_3 = logical(mask_3); % seeded region growing + convex hull
    mask_4 = logical(mask_4); % level set
    
    
    Target = [mask_2, mask_3, mask_4]; % target mask (3 methods)

    % settings zeros to append data
    Dice = [0 ; 0 ; 0]; % dice coefficient
    IoU = [0 ; 0 ; 0]; % intersection over union
    Precision = [0 ; 0 ; 0]; % precision
    Recall = [0 ; 0 ; 0]; % recall
    
    for i = 1 : 3
        if i == 1 % mask_2 : Seeded Region Growing
            fprintf('[Seeded Region Growing]\n');
        end
        if i == 2 % mask_3 : Seeded Region Growing + Convex Hull
            fprintf('[Seeded Region Growing + Convex Hull]\n');
        end
        if i == 3 % mask_4 : Level Set
            fprintf('[Level Set]\n');
        end
        % target mask split(array slicing) to do index works using for loop : Target(:,(width(bw1)*(i-1)+1:width(bw1)*i))
        T = Target(:,(width(bw1)*(i-1)+1:width(bw1)*i)); %1?겫??苑? 256, 257?겫??苑? ... 256??뼊??맄嚥?? split(width of mask)
        [dice_, iou, precis, recall] = BinaryClassify(R, T); % dice, iou, precision, recall ??④쑴沅? ?釉???땾(筌?? ?釉???삋 筌〓챷???)
        
        % Score matrix elements by index "i"
        Dice(i) = dice_; 
        IoU(i) = iou;
        Precision(i) = precis;
        Recall(i) = recall;

        PrintComparisionValue(dice_, iou, precis, recall); % print scores
    
    end
    
    ThreeDimTable(:,:,index) = horzcat(Dice, IoU, Precision, Recall); % append data to 2D tables
    
    Segmentation_Method = {'Seeded Region Growing' ; 'Seeded Region Growing + Convex Hull' ; 'Level Set'}
    
    Table_c = table(Segmentation_Method, Dice, IoU, Precision, Recall) %create #2.5 Table
    
    clearvars -except ThreeDimTable 
   
    % vars = {'img1','bw1'};
    % clear(vars{:})
end % c001, c002, c003, c006 file load and append to data "ThreeDimTable" and the end.

MeanTable = MeanOfThe4Tables(ThreeDimTable); % MeanTable
SDTable = StandardDeviationTables(ThreeDimTable, MeanTable); % SDTable
CalculatedTable = string(MeanTable)+'±'+string(SDTable) % Convert double into string / Create 'Mean ± SD' Form Table 
%%
function [dice_, iou, precis, recall] = BinaryClassify(R, T)
dice_ = dice(R, T);
iou = jaccard(R, T);

tp = sum((R == 1) & (T == 1));
fp = sum((R == 0) & (T == 1));
fn = sum((R == 1) & (T == 0));
precis = tp / (tp + fp);
recall = tp / (tp + fn);
end

function PrintComparisionValue(dice_, iou, precis, recall)
fprintf('Dice Score : %f\n', dice_);
fprintf('IoU : %f\n', iou);
fprintf('Precision : %f\n', precis);
fprintf('Recall : %f\n', recall);
fprintf('\n');
end

function MeanTable = MeanOfThe4Tables(X)
    MeanTable = (X(:,:,1)+X(:,:,2)+X(:,:,3)+X(:,:,4))./4
end


function SDTable = StandardDeviationTables(X, Y) 
    SDTable = sqrt(((X(:,:,1)-Y).^2 + (X(:,:,2)-Y).^2 + (X(:,:,3)-Y).^2 + (X(:,:,4)-Y).^2)./4) 
end % SD = sqrt(((mean-data)^2)/datacount)
