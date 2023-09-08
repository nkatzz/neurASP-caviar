
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(mc1gt_1,1,288). coords(mc1gt_1,222,225,288).  visible(mc1gt_1,288). happensAt(walking(mc1gt_1),288). orientation(mc1gt_3,135,288). coords(mc1gt_3,263,146,288).  visible(mc1gt_3,288). happensAt(walking(mc1gt_3),288).
orientation(mc1gt_1,3,289). coords(mc1gt_1,222,228,289).  visible(mc1gt_1,289). happensAt(walking(mc1gt_1),289). orientation(mc1gt_3,135,289). coords(mc1gt_3,264,148,289).  visible(mc1gt_3,289). happensAt(walking(mc1gt_3),289).
orientation(mc1gt_1,0,290). coords(mc1gt_1,222,230,290).  visible(mc1gt_1,290). happensAt(walking(mc1gt_1),290). orientation(mc1gt_3,135,290). coords(mc1gt_3,264,150,290).  visible(mc1gt_3,290). happensAt(walking(mc1gt_3),290).
orientation(mc1gt_1,171,291). coords(mc1gt_1,222,232,291).  visible(mc1gt_1,291). happensAt(walking(mc1gt_1),291). orientation(mc1gt_3,135,291). coords(mc1gt_3,265,153,291).  visible(mc1gt_3,291). happensAt(walking(mc1gt_3),291).
orientation(mc1gt_1,172,292). coords(mc1gt_1,222,235,292).  visible(mc1gt_1,292). happensAt(walking(mc1gt_1),292). orientation(mc1gt_3,135,292). coords(mc1gt_3,269,158,292).  visible(mc1gt_3,292). happensAt(walking(mc1gt_3),292).
orientation(mc1gt_1,174,293). coords(mc1gt_1,222,237,293).  visible(mc1gt_1,293). happensAt(walking(mc1gt_1),293). orientation(mc1gt_3,130,293). coords(mc1gt_3,271,162,293).  visible(mc1gt_3,293). happensAt(walking(mc1gt_3),293).
orientation(mc1gt_1,0,294). coords(mc1gt_1,222,239,294).  visible(mc1gt_1,294). happensAt(walking(mc1gt_1),294). orientation(mc1gt_3,125,294). coords(mc1gt_3,274,166,294).  visible(mc1gt_3,294). happensAt(walking(mc1gt_3),294).
orientation(mc1gt_1,3,295). coords(mc1gt_1,222,239,295).  visible(mc1gt_1,295). happensAt(walking(mc1gt_1),295). orientation(mc1gt_3,123,295). coords(mc1gt_3,276,169,295).  visible(mc1gt_3,295). happensAt(walking(mc1gt_3),295).
orientation(mc1gt_1,178,296). coords(mc1gt_1,222,240,296).  visible(mc1gt_1,296). happensAt(walking(mc1gt_1),296). orientation(mc1gt_3,124,296). coords(mc1gt_3,278,172,296).  visible(mc1gt_3,296). happensAt(walking(mc1gt_3),296).
orientation(mc1gt_1,3,297). coords(mc1gt_1,222,240,297).  visible(mc1gt_1,297). happensAt(walking(mc1gt_1),297). orientation(mc1gt_3,124,297). coords(mc1gt_3,280,174,297).  visible(mc1gt_3,297). happensAt(walking(mc1gt_3),297).
orientation(mc1gt_1,6,298). coords(mc1gt_1,222,241,298).  visible(mc1gt_1,298). happensAt(walking(mc1gt_1),298). orientation(mc1gt_3,131,298). coords(mc1gt_3,281,176,298).  visible(mc1gt_3,298). happensAt(walking(mc1gt_3),298).
orientation(mc1gt_1,0,299). coords(mc1gt_1,222,243,299).  visible(mc1gt_1,299). happensAt(walking(mc1gt_1),299). orientation(mc1gt_3,134,299). coords(mc1gt_3,283,176,299).  visible(mc1gt_3,299). happensAt(walking(mc1gt_3),299).
orientation(mc1gt_1,1,300). coords(mc1gt_1,222,245,300).  visible(mc1gt_1,300). happensAt(walking(mc1gt_1),300). orientation(mc1gt_3,135,300). coords(mc1gt_3,283,177,300).  visible(mc1gt_3,300). happensAt(walking(mc1gt_3),300).
orientation(mc1gt_1,5,301). coords(mc1gt_1,223,248,301).  visible(mc1gt_1,301). happensAt(walking(mc1gt_1),301). orientation(mc1gt_3,135,301). coords(mc1gt_3,284,179,301).  visible(mc1gt_3,301). happensAt(walking(mc1gt_3),301).
orientation(mc1gt_1,14,302). coords(mc1gt_1,223,252,302).  visible(mc1gt_1,302). happensAt(walking(mc1gt_1),302). orientation(mc1gt_3,135,302). coords(mc1gt_3,285,181,302).  visible(mc1gt_3,302). happensAt(walking(mc1gt_3),302).
orientation(mc1gt_1,5,303). coords(mc1gt_1,223,255,303).  visible(mc1gt_1,303). happensAt(walking(mc1gt_1),303). orientation(mc1gt_3,135,303). coords(mc1gt_3,286,182,303).  visible(mc1gt_3,303). happensAt(walking(mc1gt_3),303).
orientation(mc1gt_1,7,304). coords(mc1gt_1,224,259,304).  visible(mc1gt_1,304). happensAt(walking(mc1gt_1),304). orientation(mc1gt_3,135,304). coords(mc1gt_3,287,184,304).  visible(mc1gt_3,304). happensAt(walking(mc1gt_3),304).
orientation(mc1gt_1,13,305). coords(mc1gt_1,224,262,305).  visible(mc1gt_1,305). happensAt(walking(mc1gt_1),305). orientation(mc1gt_3,135,305). coords(mc1gt_3,287,185,305).  visible(mc1gt_3,305). happensAt(walking(mc1gt_3),305).
orientation(mc1gt_1,12,306). coords(mc1gt_1,224,263,306).  visible(mc1gt_1,306). happensAt(walking(mc1gt_1),306). orientation(mc1gt_3,152,306). coords(mc1gt_3,291,188,306).  visible(mc1gt_3,306). happensAt(walking(mc1gt_3),306).
orientation(mc1gt_1,8,307). coords(mc1gt_1,225,264,307).  visible(mc1gt_1,307). happensAt(walking(mc1gt_1),307). orientation(mc1gt_3,160,307). coords(mc1gt_3,293,192,307).  visible(mc1gt_3,307). happensAt(walking(mc1gt_3),307).
orientation(mc1gt_1,15,308). coords(mc1gt_1,226,265,308).  visible(mc1gt_1,308). happensAt(walking(mc1gt_1),308). orientation(mc1gt_3,172,308). coords(mc1gt_3,294,196,308).  visible(mc1gt_3,308). happensAt(walking(mc1gt_3),308).
orientation(mc1gt_1,14,309). coords(mc1gt_1,226,267,309).  visible(mc1gt_1,309). happensAt(walking(mc1gt_1),309). orientation(mc1gt_3,3,309). coords(mc1gt_3,297,199,309).  visible(mc1gt_3,309). happensAt(walking(mc1gt_3),309).
orientation(mc1gt_1,20,310). coords(mc1gt_1,227,267,310).  visible(mc1gt_1,310). happensAt(walking(mc1gt_1),310). orientation(mc1gt_3,1,310). coords(mc1gt_3,298,201,310).  visible(mc1gt_3,310). happensAt(walking(mc1gt_3),310).
orientation(mc1gt_1,19,311). coords(mc1gt_1,227,267,311).  visible(mc1gt_1,311). happensAt(walking(mc1gt_1),311). orientation(mc1gt_3,6,311). coords(mc1gt_3,299,203,311).  visible(mc1gt_3,311). happensAt(walking(mc1gt_3),311).
orientation(mc1gt_1,32,312). coords(mc1gt_1,227,267,312).  visible(mc1gt_1,312). happensAt(walking(mc1gt_1),312). orientation(mc1gt_3,178,312). coords(mc1gt_3,300,204,312).  visible(mc1gt_3,312). happensAt(walking(mc1gt_3),312).
#show holdsAt/2.