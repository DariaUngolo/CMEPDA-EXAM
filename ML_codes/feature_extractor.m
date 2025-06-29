
function [Means, Stds, Volumes] = feature_extractor(folder_path, atlas_file, atlas_txt)
    % FEATURE_EXTRACTOR Extracts mean, standard deviation (std) and volume for each ROI from NIfTI images.
    %
    % Description:
    %   This function extracts the mean, the standard deviation and the volume of voxel intensities
    %   for each region of interest (ROI) in the brain as defined in an atlas.
    %   The image files should be in NIfTI format (.nii/.nii.gz) and should be located
    %   in the specified folder. The function uses an atlas file (.nii) to segment the
    %   brain into regions and then computes the desired statistics for each ROI.
    %
    % Inputs:
    %   - folder_path : string. Path to the folder containing NIfTI image files.
    %   - atlas_file  : string. Path to the atlas NIfTI file used for segmentation.
    %   - atlas_txt   : string. Path to a text file with ROI IDs and names (format: ID<TAB>Name).
    %
    % Outputs:
    %   - A CSV file for each ROI, containing the mean and standard deviation values.
    %
    % Example:
    %   feature_extractor('path/to/folder', 'path/to/atlas.nii', 'path/to/atlas_labels.txt', 'results_prefix');

    % Intensity threshold to ignore very small voxel values (e.g., background)
    INTENSITY_THRESHOLD = 1e-6;

    %% 1. Load images from the folder
    nifti_files = dir(fullfile(folder_path, '*.nii*'));
    image_filepaths = string(fullfile({nifti_files.folder}, {nifti_files.name}));
    num_images = numel(image_filepaths);

    % Check if any NIfTI files were found
    if num_images == 0
        error('No NIfTI files found in the folder: %s', folder_path);
    end

    %% 2. Load the atlas
    atlas_data = double(niftiread(atlas_file));
    atlas_size = size(atlas_data);

    %% 3. Load ROI data
    atlas_reading = fopen(atlas_txt, 'r');
    roi_data = textscan(atlas_reading, '%d%s', 'Delimiter', '\t');
    fclose(atlas_reading);

    roi_ids = roi_data{1};   % ROI IDs (numeric)
    roi_names = roi_data{2}; % ROI names (strings)
    num_rois = numel(roi_ids);

    %% 4. Pre-calculate ROI masks
    % Create logical masks for each ROI, which will later be used to extract voxel values
    roi_masks = false([atlas_size, num_rois]);
    for j = 1:num_rois
        roi_masks(:,:,:,j) = (atlas_data == roi_ids(j));
    end

    %% 4.extra Load image header to get voxel size
    img_header = niftiinfo(image_filepaths(1));  % Get the header of the first image to get voxel size  
    voxel_size = img_header.PixelDimensions;  % Voxel size (in mm) for each of axis x, y, z  
    voxel_volume = prod(voxel_size);  % Volume of a voxel in mm^3

   
    %% 5. Pre-allocate results
    % Initialize matrices to store the mean and standard deviation values for each image and ROI
    Means = NaN(num_images, num_rois);
    Stds = NaN(num_images, num_rois);
    Volumes = NaN(num_images, num_rois);

    %% 6. Feature extraction
    % Loop through all images and calculate the features for each ROI
    for i = 1:num_images
        % Load the current image as a double precision array
        img = double(niftiread(image_filepaths(i)));

        % Loop through all ROIs and compute the features for each ROI
        for j = 1:num_rois
            % Extract the voxel intensities corresponding to the current ROI
            voxels = img(roi_masks(:,:,:,j));

            % Remove very small values (background) by applying the intensity threshold
            voxels = voxels(voxels > INTENSITY_THRESHOLD);

            % If there are valid voxels, calculate the features
            if ~isempty(voxels)
                Means(i,j) = mean(voxels);
                Stds(i,j) = std(voxels);

                % Calculate volume (number of voxels * voxel volume)
                num_voxels = numel(voxels);  % Count number of voxels in the ROI  
                Volumes(i,j) = num_voxels * voxel_volume;  % Calculate volume in mm^3  
            end
        end
    end

    %% 7. Create separate tables for means and standard deviations
    % Prepare table column names fro the features for each ROI
    image_filepaths = regexprep(image_filepaths, 'smwc1', ' '); %replace occurrences of the string 'smwc1' in the variable image_filepaths with a blank space ' '.
    [~, base_names, ~] = cellfun(@fileparts, cellstr(image_filepaths), 'UniformOutput', false); %cellfun applies the fileparts function to each element of the image_filepaths array.
                                                                                                %fileparts extracts the parts of the file paths, such as the folder path, base file name, and file extension. 
                                                                                                % In this case, only the base file name (base_names) is being used.
    img_names = string(base_names)';
    mean_colnames = strcat("Mean_", roi_names'); %concatenation of strings
    std_colnames = strcat("Std_", roi_names');
    vol_colnames = strcat("Volume_", roi_names');  

    % Create tables for the features
    MeanTable = array2table(Means, 'VariableNames', mean_colnames);
    MeanTable = addvars(MeanTable, img_names, 'Before', 1, 'NewVariableNames', 'Image');

    StdTable = array2table(Stds, 'VariableNames', std_colnames);
    StdTable = addvars(StdTable, img_names, 'Before', 1, 'NewVariableNames', 'Image');

    VolTable = array2table(Volumes, 'VariableNames', vol_colnames);  
    VolTable = addvars(VolTable, img_names, 'Before', 1, 'NewVariableNames', 'Image'); 

    
end

