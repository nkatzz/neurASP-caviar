
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

	orientation(mc1gt_0,168,288). coords(mc1gt_0,191,213,288).  visible(mc1gt_0,288). happensAt(walking(mc1gt_0),288). orientation(mc1gt_3,135,288). coords(mc1gt_3,263,146,288).  visible(mc1gt_3,288). happensAt(walking(mc1gt_3),288).
orientation(mc1gt_0,168,289). coords(mc1gt_0,191,214,289).  visible(mc1gt_0,289). happensAt(walking(mc1gt_0),289). orientation(mc1gt_3,135,289). coords(mc1gt_3,264,148,289).  visible(mc1gt_3,289). happensAt(walking(mc1gt_3),289).
orientation(mc1gt_0,168,290). coords(mc1gt_0,195,216,290).  visible(mc1gt_0,290). happensAt(walking(mc1gt_0),290). orientation(mc1gt_3,135,290). coords(mc1gt_3,264,150,290).  visible(mc1gt_3,290). happensAt(walking(mc1gt_3),290).
orientation(mc1gt_0,174,291). coords(mc1gt_0,196,217,291).  visible(mc1gt_0,291). happensAt(walking(mc1gt_0),291). orientation(mc1gt_3,135,291). coords(mc1gt_3,265,153,291).  visible(mc1gt_3,291). happensAt(walking(mc1gt_3),291).
orientation(mc1gt_0,179,292). coords(mc1gt_0,197,218,292).  visible(mc1gt_0,292). happensAt(walking(mc1gt_0),292). orientation(mc1gt_3,135,292). coords(mc1gt_3,269,158,292).  visible(mc1gt_3,292). happensAt(walking(mc1gt_3),292).
orientation(mc1gt_0,1,293). coords(mc1gt_0,197,219,293).  visible(mc1gt_0,293). happensAt(walking(mc1gt_0),293). orientation(mc1gt_3,130,293). coords(mc1gt_3,271,162,293).  visible(mc1gt_3,293). happensAt(walking(mc1gt_3),293).
orientation(mc1gt_0,1,294). coords(mc1gt_0,198,220,294).  visible(mc1gt_0,294). happensAt(walking(mc1gt_0),294). orientation(mc1gt_3,125,294). coords(mc1gt_3,274,166,294).  visible(mc1gt_3,294). happensAt(walking(mc1gt_3),294).
orientation(mc1gt_0,1,295). coords(mc1gt_0,199,221,295).  visible(mc1gt_0,295). happensAt(walking(mc1gt_0),295). orientation(mc1gt_3,123,295). coords(mc1gt_3,276,169,295).  visible(mc1gt_3,295). happensAt(walking(mc1gt_3),295).
orientation(mc1gt_0,1,296). coords(mc1gt_0,199,223,296).  visible(mc1gt_0,296). happensAt(walking(mc1gt_0),296). orientation(mc1gt_3,124,296). coords(mc1gt_3,278,172,296).  visible(mc1gt_3,296). happensAt(walking(mc1gt_3),296).
orientation(mc1gt_0,1,297). coords(mc1gt_0,199,224,297).  visible(mc1gt_0,297). happensAt(walking(mc1gt_0),297). orientation(mc1gt_3,124,297). coords(mc1gt_3,280,174,297).  visible(mc1gt_3,297). happensAt(walking(mc1gt_3),297).
orientation(mc1gt_0,1,298). coords(mc1gt_0,199,225,298).  visible(mc1gt_0,298). happensAt(walking(mc1gt_0),298). orientation(mc1gt_3,131,298). coords(mc1gt_3,281,176,298).  visible(mc1gt_3,298). happensAt(walking(mc1gt_3),298).
orientation(mc1gt_0,1,299). coords(mc1gt_0,199,229,299).  visible(mc1gt_0,299). happensAt(walking(mc1gt_0),299). orientation(mc1gt_3,134,299). coords(mc1gt_3,283,176,299).  visible(mc1gt_3,299). happensAt(walking(mc1gt_3),299).
orientation(mc1gt_0,1,300). coords(mc1gt_0,199,231,300).  visible(mc1gt_0,300). happensAt(walking(mc1gt_0),300). orientation(mc1gt_3,135,300). coords(mc1gt_3,283,177,300).  visible(mc1gt_3,300). happensAt(walking(mc1gt_3),300).
orientation(mc1gt_0,1,301). coords(mc1gt_0,200,231,301).  visible(mc1gt_0,301). happensAt(walking(mc1gt_0),301). orientation(mc1gt_3,135,301). coords(mc1gt_3,284,179,301).  visible(mc1gt_3,301). happensAt(walking(mc1gt_3),301).
orientation(mc1gt_0,1,302). coords(mc1gt_0,200,232,302).  visible(mc1gt_0,302). happensAt(walking(mc1gt_0),302). orientation(mc1gt_3,135,302). coords(mc1gt_3,285,181,302).  visible(mc1gt_3,302). happensAt(walking(mc1gt_3),302).
orientation(mc1gt_0,1,303). coords(mc1gt_0,201,233,303).  visible(mc1gt_0,303). happensAt(walking(mc1gt_0),303). orientation(mc1gt_3,135,303). coords(mc1gt_3,286,182,303).  visible(mc1gt_3,303). happensAt(walking(mc1gt_3),303).
orientation(mc1gt_0,1,304). coords(mc1gt_0,202,233,304).  visible(mc1gt_0,304). happensAt(walking(mc1gt_0),304). orientation(mc1gt_3,135,304). coords(mc1gt_3,287,184,304).  visible(mc1gt_3,304). happensAt(walking(mc1gt_3),304).
orientation(mc1gt_0,4,305). coords(mc1gt_0,202,234,305).  visible(mc1gt_0,305). happensAt(walking(mc1gt_0),305). orientation(mc1gt_3,135,305). coords(mc1gt_3,287,185,305).  visible(mc1gt_3,305). happensAt(walking(mc1gt_3),305).
orientation(mc1gt_0,4,306). coords(mc1gt_0,202,234,306).  visible(mc1gt_0,306). happensAt(walking(mc1gt_0),306). orientation(mc1gt_3,152,306). coords(mc1gt_3,291,188,306).  visible(mc1gt_3,306). happensAt(walking(mc1gt_3),306).
orientation(mc1gt_0,4,307). coords(mc1gt_0,201,235,307).  visible(mc1gt_0,307). happensAt(walking(mc1gt_0),307). orientation(mc1gt_3,160,307). coords(mc1gt_3,293,192,307).  visible(mc1gt_3,307). happensAt(walking(mc1gt_3),307).
orientation(mc1gt_0,10,308). coords(mc1gt_0,201,237,308).  visible(mc1gt_0,308). happensAt(walking(mc1gt_0),308). orientation(mc1gt_3,172,308). coords(mc1gt_3,294,196,308).  visible(mc1gt_3,308). happensAt(walking(mc1gt_3),308).
orientation(mc1gt_0,10,309). coords(mc1gt_0,202,238,309).  visible(mc1gt_0,309). happensAt(walking(mc1gt_0),309). orientation(mc1gt_3,3,309). coords(mc1gt_3,297,199,309).  visible(mc1gt_3,309). happensAt(walking(mc1gt_3),309).
orientation(mc1gt_0,10,310). coords(mc1gt_0,202,242,310).  visible(mc1gt_0,310). happensAt(walking(mc1gt_0),310). orientation(mc1gt_3,1,310). coords(mc1gt_3,298,201,310).  visible(mc1gt_3,310). happensAt(walking(mc1gt_3),310).
orientation(mc1gt_0,10,311). coords(mc1gt_0,204,244,311).  visible(mc1gt_0,311). happensAt(walking(mc1gt_0),311). orientation(mc1gt_3,6,311). coords(mc1gt_3,299,203,311).  visible(mc1gt_3,311). happensAt(walking(mc1gt_3),311).
orientation(mc1gt_0,10,312). coords(mc1gt_0,206,247,312).  visible(mc1gt_0,312). happensAt(walking(mc1gt_0),312). orientation(mc1gt_3,178,312). coords(mc1gt_3,300,204,312).  visible(mc1gt_3,312). happensAt(walking(mc1gt_3),312).
#show holdsAt/2.