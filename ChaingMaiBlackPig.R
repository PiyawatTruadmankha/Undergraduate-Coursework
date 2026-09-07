# ==============================================================================
# ภาคผนวก ค: ชุดคำสั่งโปรแกรม R สำหรับการวิเคราะห์ข้อมูลและการสร้างตัวแบบพยากรณ์
# ==============================================================================

# ------------------------------------------------------------------------------
# ส่วนที่ 1: การนำเข้าและสำรวจข้อมูลเบื้องต้น (Data Import & Exploration)
# ------------------------------------------------------------------------------
DataBLackpigs <- read.csv("Pig80.csv")

# สรุปค่าสถิติพื้นฐาน
summary(DataBLackpigs)

# install.packages("psych")
library(psych)
describe(DataBLackpigs)

# ------------------------------------------------------------------------------
# ส่วนที่ 2: การสร้างภาพนิทัศน์ข้อมูล (Data Visualization)
# ------------------------------------------------------------------------------
# 2.1 พลอตกราฟ Boxplot เพื่อดู Outlier
par(mar = c(8, 5, 4, 2))
boxplot(DataBLackpigs, las = 2, 
        main="แผนภาพกล่องแสดงการกระจายตัวของข้อมูลน้ำหนักและสัดส่วนร่างกายของหมูดำเชียงใหม่", 
        ylab="ค่าการวัด (กิโลกรัม / เซนติเมตร)", col="lightblue")

# 2.2 ดูแนวโน้มข้อมูลด้วย Scatter Plots
library(tidyverse)
DataBLackpigs %>%
  select(Weight, Heart.Girth, Abdominal.Girth, Hip.Girth, Body.Length, Chest.Width, Abdominal.Width, Hip.Width) %>%
  pivot_longer(cols = -Weight, names_to = "Variable", values_to = "Measurement") %>%
  mutate(ColorGroup = ifelse(grepl("Girth", Variable), "Girth", "Other")) %>%
  ggplot(aes(x = Measurement, y = Weight)) +
  geom_point(aes(color = ColorGroup), size = 2.5, alpha = 0.8) +
  facet_wrap(~ Variable, scales = "free_x", ncol = 3) +
  scale_color_manual(values = c("Girth" = "skyblue3", "Other" = "palegreen4")) +
  labs(title = "Relationship between Body Weight and Various Measurements",
       y = "Body Weight (kg)", x = "Measurement (cm)") +
  theme_bw() + 
  theme(legend.position = "none", 
        strip.background = element_rect(fill = "white", color = "black"), 
        strip.text = element_text(face = "bold", size = 10))

# 2.3 ตรวจสอบการแจกแจงด้วย Histogram และ Density Curve
par(mfrow=c(3,3))
cols <- c("gray80","#7DB7CF","#7DB7CF","#7DB7CF",
          "#7ED37F","#7ED37F","#7ED37F","#7ED37F")

for(i in seq_along(DataBLackpigs)){
  hist(DataBLackpigs[[i]], probability=TRUE, col=cols[i], border="white",
       main=names(DataBLackpigs)[i], xlab=names(DataBLackpigs)[i], ylab="Density")
  lines(density(DataBLackpigs[[i]]), col="red", lwd=2)
}
par(mfrow=c(1,1)) # คืนค่าหน้าต่างกราฟ

# ------------------------------------------------------------------------------
# ส่วนที่ 3: การวิเคราะห์สหสัมพันธ์ (Correlation Analysis)
# ------------------------------------------------------------------------------
# install.packages("Hmisc")
library(Hmisc)
rcorr(as.matrix(DataBLackpigs), type = "pearson")
rcorr(as.matrix(DataBLackpigs), type = "spearman")

# พลอตกราฟ Correlation Matrix
library(corrplot)
res <- cor(DataBLackpigs)
corrplot(res, method = "circle", type="upper")

# ------------------------------------------------------------------------------
# ส่วนที่ 4: การสร้างตัวแบบถดถอยเชิงเส้นพหุคูณ (Multiple Linear Regression)
# ------------------------------------------------------------------------------
# 4.1 Full Model (Stepwise Selection)
LinearBlackPigsFull <- lm(Weight ~ Heart.Girth + Abdominal.Girth + Hip.Girth + 
                            Body.Length + Chest.Width + Abdominal.Width + Hip.Width, 
                          data = DataBLackpigs)
OptimalFullModel <- step(LinearBlackPigsFull, direction = "both")
summary(OptimalFullModel)

# ติดตั้งแพ็กเกจ (รันแค่ครั้งแรกครั้งเดียว)
install.packages("car")

# เรียกใช้งานแพ็กเกจเพื่อดึงคำสั่ง vif มาใช้
library(car)
# ตรวจสอบค่า VIF ของโมเดล
vif(OptimalFullModel)

# 4.2 Width Group Model (Stepwise Selection)
LinearBlackPigsWidth <- lm(Weight ~ Body.Length + Chest.Width + Abdominal.Width + Hip.Width, 
                           data = DataBLackpigs)
OptimalWidthModel <- step(LinearBlackPigsWidth, direction = "both")
summary(OptimalWidthModel)
# ตรวจสอบค่า VIF ของโมเดล
vif(OptimalWidthModel)

# 4.3 Girth Group Model (Stepwise Selection)
LinearBlackPigsGirth <- lm(Weight ~ Heart.Girth + Abdominal.Girth + Hip.Girth, 
                           data = DataBLackpigs)
OptimalGirthModel <- step(LinearBlackPigsGirth, direction = "both")
summary(OptimalGirthModel)
# ตรวจสอบค่า VIF ของโมเดล
vif(OptimalGirthModel)

# ------------------------------------------------------------------------------
# ส่วนที่ 5: การตรวจสอบข้อตกลงเบื้องต้นของสมการถดถอย (Assumption Checking)
# ------------------------------------------------------------------------------
library(lmtest)
library(car)
library(corrplot)

# ==============================================================================
# 5.1) ตัวแบบ Full Model
# ==============================================================================
# --- Normality Test (ตรวจสอบการแจกแจงปกติของส่วนตกค้าง) ---
shapiro.test(residuals(OptimalFullModel))
plot(OptimalFullModel, 2) # ดูกราฟ Normal Q-Q Plot

# --- Homoscedasticity Test (ตรวจสอบความคงที่ของความแปรปรวน) ---
bptest(OptimalFullModel)
plot(OptimalFullModel, 1) # ดูกราฟ Residuals vs Fitted

# --- Independence Test (ตรวจสอบความเป็นอิสระของส่วนตกค้าง) ---
durbinWatsonTest(OptimalFullModel)
plot(residuals(OptimalFullModel), type="o", main="Residuals vs Order (Full Model)")
abline(h=0, col="red", lty=2)


# --- Multicollinearity Test (ตรวจสอบสภาวะร่วมเส้นตรงพหุ) ---
vif(OptimalFullModel)
pairs(~ Heart.Girth + Abdominal.Girth + Hip.Girth + Body.Length + Chest.Width + Abdominal.Width + Hip.Width, data = DataBLackpigs)
corrplot(cor(DataBLackpigs[, c("Abdominal.Girth", "Hip.Girth", "Body.Length", "Chest.Width", "Abdominal.Width", "Hip.Width")]), method="circle", type="upper")


# ==============================================================================
# 5.2) ตัวแบบ Width Group Model
# ==============================================================================
# --- Normality Test ---
shapiro.test(residuals(OptimalWidthModel))
plot(OptimalWidthModel, 2) 

# --- Homoscedasticity Test ---
bptest(OptimalWidthModel)
plot(OptimalWidthModel, 1) 

# --- Independence Test ---
durbinWatsonTest(OptimalWidthModel)
plot(residuals(OptimalWidthModel), type="o", main="Residuals vs Order (Width Model)")
abline(h=0, col="red", lty=2)

# --- Multicollinearity Test ---
vif(OptimalWidthModel)
corrplot(cor(DataBLackpigs[, c("Body.Length", "Chest.Width", "Abdominal.Width", "Hip.Width")]), method="circle", type="upper")


# ==============================================================================
# 5.3) ตัวแบบ Girth Group Model
# ==============================================================================
# --- Normality Test ---
shapiro.test(residuals(OptimalGirthModel))
plot(OptimalGirthModel, 2) 

# --- Homoscedasticity Test ---
bptest(OptimalGirthModel)
plot(OptimalGirthModel, 1) 

# --- Independence Test ---
durbinWatsonTest(OptimalGirthModel)
plot(residuals(OptimalGirthModel), type="o", main="Residuals vs Order (Girth Model)")
abline(h=0, col="red", lty=2)

# --- Multicollinearity Test ---
vif(OptimalGirthModel)
pairs(~ Heart.Girth + Abdominal.Girth + Hip.Girth, data = DataBLackpigs)
corrplot(cor(DataBLackpigs[, c("Heart.Girth", "Abdominal.Girth", "Hip.Girth")]), method="circle", type="upper")

# ------------------------------------------------------------------------------
# ส่วนที่ 6: ทดสอบความแม่นยำของตัวแบบด้วย 10-Fold Cross-Validation
# ------------------------------------------------------------------------------
library(caret)

# 6.1 สร้างฟังก์ชันคำนวณ MAPE สำหรับใช้ใน caret
my_summary <- function(data, lev = NULL, model = NULL) {
  default_metrics <- defaultSummary(data, lev, model)
  mape <- mean(abs((data$obs - data$pred) / data$obs)) * 100
  c(default_metrics, MAPE = mape)
}

# 6.2 ตั้งค่า Train Control (10-Fold CV)
train_control <- trainControl(method = "cv", number = 10, summaryFunction = my_summary)

# 6.3 ประเมินประสิทธิภาพ Girth Group Model
set.seed(123)
cv_girth <- train(Weight ~ Abdominal.Girth + Hip.Girth,
                  data = DataBLackpigs, method = "lm", trControl = train_control)
print(cv_girth)

# 6.4 ประเมินประสิทธิภาพ Width Group Model
set.seed(123)
cv_width <- train(Weight ~ Body.Length + Chest.Width + Hip.Width,
                  data = DataBLackpigs, method = "lm", trControl = train_control)
print(cv_width)

# 6.5 ประเมินประสิทธิภาพ Full Model
set.seed(123)
cv_full <- train(Weight ~ Heart.Girth + Abdominal.Girth + Hip.Girth + Chest.Width,
                 data = DataBLackpigs, method = "lm", trControl = train_control)
print(cv_full)