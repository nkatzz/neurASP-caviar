
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

	orientation(mc1gt_0,179,292). coords(mc1gt_0,197,218,292).  visible(mc1gt_0,292). happensAt(walking(mc1gt_0),292). orientation(mc1gt_1,172,292). coords(mc1gt_1,222,235,292).  visible(mc1gt_1,292). happensAt(walking(mc1gt_1),292).
orientation(mc1gt_0,1,293). coords(mc1gt_0,197,219,293).  visible(mc1gt_0,293). happensAt(walking(mc1gt_0),293). orientation(mc1gt_1,174,293). coords(mc1gt_1,222,237,293).  visible(mc1gt_1,293). happensAt(walking(mc1gt_1),293).
orientation(mc1gt_0,1,294). coords(mc1gt_0,198,220,294).  visible(mc1gt_0,294). happensAt(walking(mc1gt_0),294). orientation(mc1gt_1,0,294). coords(mc1gt_1,222,239,294).  visible(mc1gt_1,294). happensAt(walking(mc1gt_1),294).
orientation(mc1gt_0,1,295). coords(mc1gt_0,199,221,295).  visible(mc1gt_0,295). happensAt(walking(mc1gt_0),295). orientation(mc1gt_1,3,295). coords(mc1gt_1,222,239,295).  visible(mc1gt_1,295). happensAt(walking(mc1gt_1),295).
orientation(mc1gt_0,1,296). coords(mc1gt_0,199,223,296).  visible(mc1gt_0,296). happensAt(walking(mc1gt_0),296). orientation(mc1gt_1,178,296). coords(mc1gt_1,222,240,296).  visible(mc1gt_1,296). happensAt(walking(mc1gt_1),296).
orientation(mc1gt_0,1,297). coords(mc1gt_0,199,224,297).  visible(mc1gt_0,297). happensAt(walking(mc1gt_0),297). orientation(mc1gt_1,3,297). coords(mc1gt_1,222,240,297).  visible(mc1gt_1,297). happensAt(walking(mc1gt_1),297).
orientation(mc1gt_0,1,298). coords(mc1gt_0,199,225,298).  visible(mc1gt_0,298). happensAt(walking(mc1gt_0),298). orientation(mc1gt_1,6,298). coords(mc1gt_1,222,241,298).  visible(mc1gt_1,298). happensAt(walking(mc1gt_1),298).
orientation(mc1gt_0,1,299). coords(mc1gt_0,199,229,299).  visible(mc1gt_0,299). happensAt(walking(mc1gt_0),299). orientation(mc1gt_1,0,299). coords(mc1gt_1,222,243,299).  visible(mc1gt_1,299). happensAt(walking(mc1gt_1),299).
orientation(mc1gt_0,1,300). coords(mc1gt_0,199,231,300).  visible(mc1gt_0,300). happensAt(walking(mc1gt_0),300). orientation(mc1gt_1,1,300). coords(mc1gt_1,222,245,300).  visible(mc1gt_1,300). happensAt(walking(mc1gt_1),300).
orientation(mc1gt_0,1,301). coords(mc1gt_0,200,231,301).  visible(mc1gt_0,301). happensAt(walking(mc1gt_0),301). orientation(mc1gt_1,5,301). coords(mc1gt_1,223,248,301).  visible(mc1gt_1,301). happensAt(walking(mc1gt_1),301).
orientation(mc1gt_0,1,302). coords(mc1gt_0,200,232,302).  visible(mc1gt_0,302). happensAt(walking(mc1gt_0),302). orientation(mc1gt_1,14,302). coords(mc1gt_1,223,252,302).  visible(mc1gt_1,302). happensAt(walking(mc1gt_1),302).
orientation(mc1gt_0,1,303). coords(mc1gt_0,201,233,303).  visible(mc1gt_0,303). happensAt(walking(mc1gt_0),303). orientation(mc1gt_1,5,303). coords(mc1gt_1,223,255,303).  visible(mc1gt_1,303). happensAt(walking(mc1gt_1),303).
orientation(mc1gt_0,1,304). coords(mc1gt_0,202,233,304).  visible(mc1gt_0,304). happensAt(walking(mc1gt_0),304). orientation(mc1gt_1,7,304). coords(mc1gt_1,224,259,304).  visible(mc1gt_1,304). happensAt(walking(mc1gt_1),304).
orientation(mc1gt_0,4,305). coords(mc1gt_0,202,234,305).  visible(mc1gt_0,305). happensAt(walking(mc1gt_0),305). orientation(mc1gt_1,13,305). coords(mc1gt_1,224,262,305).  visible(mc1gt_1,305). happensAt(walking(mc1gt_1),305).
orientation(mc1gt_0,4,306). coords(mc1gt_0,202,234,306).  visible(mc1gt_0,306). happensAt(walking(mc1gt_0),306). orientation(mc1gt_1,12,306). coords(mc1gt_1,224,263,306).  visible(mc1gt_1,306). happensAt(walking(mc1gt_1),306).
orientation(mc1gt_0,4,307). coords(mc1gt_0,201,235,307).  visible(mc1gt_0,307). happensAt(walking(mc1gt_0),307). orientation(mc1gt_1,8,307). coords(mc1gt_1,225,264,307).  visible(mc1gt_1,307). happensAt(walking(mc1gt_1),307).
orientation(mc1gt_0,10,308). coords(mc1gt_0,201,237,308).  visible(mc1gt_0,308). happensAt(walking(mc1gt_0),308). orientation(mc1gt_1,15,308). coords(mc1gt_1,226,265,308).  visible(mc1gt_1,308). happensAt(walking(mc1gt_1),308).
orientation(mc1gt_0,10,309). coords(mc1gt_0,202,238,309).  visible(mc1gt_0,309). happensAt(walking(mc1gt_0),309). orientation(mc1gt_1,14,309). coords(mc1gt_1,226,267,309).  visible(mc1gt_1,309). happensAt(walking(mc1gt_1),309).
orientation(mc1gt_0,10,310). coords(mc1gt_0,202,242,310).  visible(mc1gt_0,310). happensAt(walking(mc1gt_0),310). orientation(mc1gt_1,20,310). coords(mc1gt_1,227,267,310).  visible(mc1gt_1,310). happensAt(walking(mc1gt_1),310).
orientation(mc1gt_0,10,311). coords(mc1gt_0,204,244,311).  visible(mc1gt_0,311). happensAt(walking(mc1gt_0),311). orientation(mc1gt_1,19,311). coords(mc1gt_1,227,267,311).  visible(mc1gt_1,311). happensAt(walking(mc1gt_1),311).
orientation(mc1gt_0,10,312). coords(mc1gt_0,206,247,312).  visible(mc1gt_0,312). happensAt(walking(mc1gt_0),312). orientation(mc1gt_1,32,312). coords(mc1gt_1,227,267,312).  visible(mc1gt_1,312). happensAt(walking(mc1gt_1),312).
orientation(mc1gt_0,10,313). coords(mc1gt_0,208,249,313).  visible(mc1gt_0,313). happensAt(walking(mc1gt_0),313). orientation(mc1gt_1,39,313). coords(mc1gt_1,228,268,313).  visible(mc1gt_1,313). happensAt(walking(mc1gt_1),313).
orientation(mc1gt_0,10,314). coords(mc1gt_0,209,251,314).  visible(mc1gt_0,314). happensAt(walking(mc1gt_0),314). orientation(mc1gt_1,43,314). coords(mc1gt_1,228,269,314).  visible(mc1gt_1,314). happensAt(walking(mc1gt_1),314).
orientation(mc1gt_0,10,315). coords(mc1gt_0,209,252,315).  visible(mc1gt_0,315). happensAt(walking(mc1gt_0),315). orientation(mc1gt_1,49,315). coords(mc1gt_1,228,270,315).  visible(mc1gt_1,315). happensAt(walking(mc1gt_1),315).
orientation(mc1gt_0,10,316). coords(mc1gt_0,210,253,316).  visible(mc1gt_0,316). happensAt(walking(mc1gt_0),316). orientation(mc1gt_1,49,316). coords(mc1gt_1,228,272,316).  visible(mc1gt_1,316). happensAt(walking(mc1gt_1),316).
#show holdsAt/2.