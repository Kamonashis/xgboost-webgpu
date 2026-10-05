use crate::booster::Booster;
use crate::tree::node::{Tree, TreeNode};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbRoot {
    pub version: Vec<u32>,
    pub learner: XgbLearner,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbLearner {
    pub generic_param: serde_json::Value,
    pub gradient_booster: XgbGradientBooster,
    pub objective: XgbObjective,
    pub learner_model_param: XgbLearnerModelParam,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbObjective {
    pub name: String,
    #[serde(default)]
    pub reg_loss_param: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbLearnerModelParam {
    pub base_score: String,
    pub num_feature: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbGradientBooster {
    pub name: String,
    pub model: XgbModel,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbModel {
    pub gbtree_model_param: XgbGbtreeModelParam,
    pub trees: Vec<XgbTree>,
    pub tree_info: Vec<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbGbtreeModelParam {
    pub num_trees: String,
    pub size_leaf_vector: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbTree {
    pub id: i32,
    pub tree_param: XgbTreeParam,
    pub loss_changes: Vec<f32>,
    pub sum_hessian: Vec<f32>,
    pub base_weights: Vec<f32>,
    pub left_children: Vec<i32>,
    pub right_children: Vec<i32>,
    pub parents: Vec<i32>,
    pub split_indices: Vec<u32>,
    pub split_conditions: Vec<f32>,
    pub default_left: Vec<i32>,
    pub split_type: Vec<i32>,
    pub categories: Vec<i32>,
    pub categories_nodes: Vec<i32>,
    pub categories_segments: Vec<i32>,
    pub categories_sizes: Vec<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct XgbTreeParam {
    pub num_nodes: String,
    pub num_feature: String,
    pub size_leaf_vector: String,
}

impl Booster {
    /// Serializes model to standard XGBoost JSON string.
    pub fn to_xgboost_json(&self) -> Result<String, String> {
        let num_trees = self.trees.len();
        let mut xgb_trees = Vec::with_capacity(num_trees);

        for (tree_id, tree) in self.trees.iter().enumerate() {
            let num_nodes = tree.nodes.len();
            let mut loss_changes = Vec::with_capacity(num_nodes);
            let mut sum_hessian = Vec::with_capacity(num_nodes);
            let mut base_weights = Vec::with_capacity(num_nodes);
            let mut left_children = Vec::with_capacity(num_nodes);
            let mut right_children = Vec::with_capacity(num_nodes);
            let mut parents = vec![0i32; num_nodes];
            let mut split_indices = Vec::with_capacity(num_nodes);
            let mut split_conditions = Vec::with_capacity(num_nodes);
            let mut default_left = Vec::with_capacity(num_nodes);
            let mut split_type = Vec::with_capacity(num_nodes);

            for (idx, node) in tree.nodes.iter().enumerate() {
                loss_changes.push(node.gain);
                sum_hessian.push(node.cover);
                base_weights.push(if node.is_leaf { node.leaf_value } else { 0.0 });

                if node.is_leaf {
                    left_children.push(-1);
                    right_children.push(-1);
                    split_indices.push(0);
                    split_conditions.push(node.leaf_value);
                    default_left.push(0);
                    split_type.push(0);
                } else {
                    let l = node.left_child.unwrap_or(0) as i32;
                    let r = node.right_child.unwrap_or(0) as i32;
                    left_children.push(l);
                    right_children.push(r);
                    if (l as usize) < num_nodes {
                        parents[l as usize] = idx as i32;
                    }
                    if (r as usize) < num_nodes {
                        parents[r as usize] = idx as i32;
                    }
                    split_indices.push(node.split_feature as u32);
                    split_conditions.push(node.split_value);
                    default_left.push(if node.default_left { 1 } else { 0 });
                    split_type.push(0);
                }
            }

            xgb_trees.push(XgbTree {
                id: tree_id as i32,
                tree_param: XgbTreeParam {
                    num_nodes: num_nodes.to_string(),
                    num_feature: self.num_features.to_string(),
                    size_leaf_vector: "0".to_string(),
                },
                loss_changes,
                sum_hessian,
                base_weights,
                left_children,
                right_children,
                parents,
                split_indices,
                split_conditions,
                default_left,
                split_type,
                categories: vec![],
                categories_nodes: vec![],
                categories_segments: vec![],
                categories_sizes: vec![],
            });
        }

        let root = XgbRoot {
            version: vec![2, 0, 0],
            learner: XgbLearner {
                generic_param: serde_json::json!({}),
                gradient_booster: XgbGradientBooster {
                    name: "gbtree".to_string(),
                    model: XgbModel {
                        gbtree_model_param: XgbGbtreeModelParam {
                            num_trees: num_trees.to_string(),
                            size_leaf_vector: "0".to_string(),
                        },
                        trees: xgb_trees,
                        tree_info: vec![0; num_trees],
                    },
                },
                objective: XgbObjective {
                    name: self.params.objective.clone(),
                    reg_loss_param: None,
                },
                learner_model_param: XgbLearnerModelParam {
                    base_score: self.base_score.to_string(),
                    num_feature: self.num_features.to_string(),
                },
            },
        };

        serde_json::to_string_pretty(&root)
            .map_err(|e| format!("Failed to serialize XGBoost JSON: {:?}", e))
    }

    /// Saves model directly to a JSON file.
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let json_str = self.to_xgboost_json()?;
        let mut file = File::create(path).map_err(|e| format!("Failed to create file: {:?}", e))?;
        file.write_all(json_str.as_bytes())
            .map_err(|e| format!("Failed to write file: {:?}", e))?;
        Ok(())
    }

    /// Loads a model from an XGBoost JSON string.
    pub fn from_xgboost_json(json_str: &str) -> Result<Self, String> {
        let root: XgbRoot = serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to deserialize XGBoost JSON: {:?}", e))?;

        let num_features: usize = root
            .learner
            .learner_model_param
            .num_feature
            .parse()
            .map_err(|e| format!("Failed to parse num_feature: {:?}", e))?;

        let base_score: f32 = root
            .learner
            .learner_model_param
            .base_score
            .parse()
            .unwrap_or(0.5);

        let mut trees = Vec::new();
        for xgb_tree in root.learner.gradient_booster.model.trees {
            let num_nodes = xgb_tree.left_children.len();
            let mut nodes = Vec::with_capacity(num_nodes);

            for i in 0..num_nodes {
                let left = xgb_tree.left_children[i];
                let right = xgb_tree.right_children[i];
                let is_leaf = left == -1 && right == -1;

                if is_leaf {
                    let leaf_val = xgb_tree.split_conditions[i];
                    let cover = xgb_tree.sum_hessian.get(i).copied().unwrap_or(1.0);
                    nodes.push(TreeNode::new_leaf(leaf_val, cover));
                } else {
                    let split_feat = xgb_tree.split_indices[i] as usize;
                    let split_cond = xgb_tree.split_conditions[i];
                    let def_left = xgb_tree.default_left[i] == 1;
                    let gain = xgb_tree.loss_changes[i];
                    let cover = xgb_tree.sum_hessian.get(i).copied().unwrap_or(1.0);

                    let mut node = TreeNode::new_split(split_feat, 0, split_cond, def_left, gain, cover);
                    node.left_child = Some(left as usize);
                    node.right_child = Some(right as usize);
                    nodes.push(node);
                }
            }

            trees.push(Tree { nodes });
        }

        let mut params = crate::booster::BoosterParams::new();
        params.objective = root.learner.objective.name;
        params.base_score = Some(base_score);

        let feature_names = (0..num_features).map(|j| format!("f{j}")).collect();

        Ok(Self {
            params,
            trees,
            base_score,
            num_features,
            feature_names,
        })
    }

    /// Loads model from a JSON file.
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open file: {:?}", e))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("Failed to read file: {:?}", e))?;
        Self::from_xgboost_json(&contents)
    }
}
