
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

	orientation(mc1gt_1,3,297). coords(mc1gt_1,222,240,297).  visible(mc1gt_1,297). happensAt(walking(mc1gt_1),297). orientation(mc1gt_2,168,297). coords(mc1gt_2,194,187,297).  visible(mc1gt_2,297). happensAt(walking(mc1gt_2),297).
orientation(mc1gt_1,6,298). coords(mc1gt_1,222,241,298).  visible(mc1gt_1,298). happensAt(walking(mc1gt_1),298). orientation(mc1gt_2,170,298). coords(mc1gt_2,196,188,298).  visible(mc1gt_2,298). happensAt(walking(mc1gt_2),298).
orientation(mc1gt_1,0,299). coords(mc1gt_1,222,243,299).  visible(mc1gt_1,299). happensAt(walking(mc1gt_1),299). orientation(mc1gt_2,172,299). coords(mc1gt_2,200,190,299).  visible(mc1gt_2,299). happensAt(walking(mc1gt_2),299).
orientation(mc1gt_1,1,300). coords(mc1gt_1,222,245,300).  visible(mc1gt_1,300). happensAt(walking(mc1gt_1),300). orientation(mc1gt_2,170,300). coords(mc1gt_2,202,192,300).  visible(mc1gt_2,300). happensAt(walking(mc1gt_2),300).
orientation(mc1gt_1,5,301). coords(mc1gt_1,223,248,301).  visible(mc1gt_1,301). happensAt(walking(mc1gt_1),301). orientation(mc1gt_2,173,301). coords(mc1gt_2,205,193,301).  visible(mc1gt_2,301). happensAt(walking(mc1gt_2),301).
orientation(mc1gt_1,14,302). coords(mc1gt_1,223,252,302).  visible(mc1gt_1,302). happensAt(walking(mc1gt_1),302). orientation(mc1gt_2,176,302). coords(mc1gt_2,207,195,302).  visible(mc1gt_2,302). happensAt(walking(mc1gt_2),302).
orientation(mc1gt_1,5,303). coords(mc1gt_1,223,255,303).  visible(mc1gt_1,303). happensAt(walking(mc1gt_1),303). orientation(mc1gt_2,178,303). coords(mc1gt_2,208,196,303).  visible(mc1gt_2,303). happensAt(walking(mc1gt_2),303).
orientation(mc1gt_1,7,304). coords(mc1gt_1,224,259,304).  visible(mc1gt_1,304). happensAt(walking(mc1gt_1),304). orientation(mc1gt_2,178,304). coords(mc1gt_2,211,197,304).  visible(mc1gt_2,304). happensAt(walking(mc1gt_2),304).
orientation(mc1gt_1,13,305). coords(mc1gt_1,224,262,305).  visible(mc1gt_1,305). happensAt(walking(mc1gt_1),305). orientation(mc1gt_2,178,305). coords(mc1gt_2,211,199,305).  visible(mc1gt_2,305). happensAt(walking(mc1gt_2),305).
orientation(mc1gt_1,12,306). coords(mc1gt_1,224,263,306).  visible(mc1gt_1,306). happensAt(walking(mc1gt_1),306). orientation(mc1gt_2,178,306). coords(mc1gt_2,212,200,306).  visible(mc1gt_2,306). happensAt(walking(mc1gt_2),306).
orientation(mc1gt_1,8,307). coords(mc1gt_1,225,264,307).  visible(mc1gt_1,307). happensAt(walking(mc1gt_1),307). orientation(mc1gt_2,178,307). coords(mc1gt_2,213,200,307).  visible(mc1gt_2,307). happensAt(walking(mc1gt_2),307).
orientation(mc1gt_1,15,308). coords(mc1gt_1,226,265,308).  visible(mc1gt_1,308). happensAt(walking(mc1gt_1),308). orientation(mc1gt_2,178,308). coords(mc1gt_2,213,202,308).  visible(mc1gt_2,308). happensAt(walking(mc1gt_2),308).
orientation(mc1gt_1,14,309). coords(mc1gt_1,226,267,309).  visible(mc1gt_1,309). happensAt(walking(mc1gt_1),309). orientation(mc1gt_2,178,309). coords(mc1gt_2,214,203,309).  visible(mc1gt_2,309). happensAt(walking(mc1gt_2),309).
orientation(mc1gt_1,20,310). coords(mc1gt_1,227,267,310).  visible(mc1gt_1,310). happensAt(walking(mc1gt_1),310). orientation(mc1gt_2,178,310). coords(mc1gt_2,214,204,310).  visible(mc1gt_2,310). happensAt(walking(mc1gt_2),310).
orientation(mc1gt_1,19,311). coords(mc1gt_1,227,267,311).  visible(mc1gt_1,311). happensAt(walking(mc1gt_1),311). orientation(mc1gt_2,178,311). coords(mc1gt_2,215,206,311).  visible(mc1gt_2,311). happensAt(walking(mc1gt_2),311).
orientation(mc1gt_1,32,312). coords(mc1gt_1,227,267,312).  visible(mc1gt_1,312). happensAt(walking(mc1gt_1),312). orientation(mc1gt_2,178,312). coords(mc1gt_2,217,208,312).  visible(mc1gt_2,312). happensAt(walking(mc1gt_2),312).
orientation(mc1gt_1,39,313). coords(mc1gt_1,228,268,313).  visible(mc1gt_1,313). happensAt(walking(mc1gt_1),313). orientation(mc1gt_2,178,313). coords(mc1gt_2,217,211,313).  visible(mc1gt_2,313). happensAt(walking(mc1gt_2),313).
orientation(mc1gt_1,43,314). coords(mc1gt_1,228,269,314).  visible(mc1gt_1,314). happensAt(walking(mc1gt_1),314). orientation(mc1gt_2,178,314). coords(mc1gt_2,218,214,314).  visible(mc1gt_2,314). happensAt(walking(mc1gt_2),314).
orientation(mc1gt_1,49,315). coords(mc1gt_1,228,270,315).  visible(mc1gt_1,315). happensAt(walking(mc1gt_1),315). orientation(mc1gt_2,178,315). coords(mc1gt_2,219,218,315).  visible(mc1gt_2,315). happensAt(walking(mc1gt_2),315).
orientation(mc1gt_1,49,316). coords(mc1gt_1,228,272,316).  visible(mc1gt_1,316). happensAt(walking(mc1gt_1),316). orientation(mc1gt_2,178,316). coords(mc1gt_2,221,220,316).  visible(mc1gt_2,316). happensAt(walking(mc1gt_2),316).
orientation(mc1gt_1,32,317). coords(mc1gt_1,228,275,317).  visible(mc1gt_1,317). happensAt(walking(mc1gt_1),317). orientation(mc1gt_2,178,317). coords(mc1gt_2,222,221,317).  visible(mc1gt_2,317). happensAt(walking(mc1gt_2),317).
orientation(mc1gt_1,28,318). coords(mc1gt_1,229,277,318).  visible(mc1gt_1,318). happensAt(walking(mc1gt_1),318). orientation(mc1gt_2,178,318). coords(mc1gt_2,223,222,318).  visible(mc1gt_2,318). happensAt(walking(mc1gt_2),318).
orientation(mc1gt_1,30,319). coords(mc1gt_1,232,279,319).  visible(mc1gt_1,319). happensAt(walking(mc1gt_1),319). orientation(mc1gt_2,178,319). coords(mc1gt_2,224,223,319).  visible(mc1gt_2,319). happensAt(walking(mc1gt_2),319).
orientation(mc1gt_1,17,320). coords(mc1gt_1,235,280,320).  visible(mc1gt_1,320). happensAt(walking(mc1gt_1),320). orientation(mc1gt_2,0,320). coords(mc1gt_2,224,223,320).  visible(mc1gt_2,320). happensAt(walking(mc1gt_2),320).
orientation(mc1gt_1,48,321). coords(mc1gt_1,236,280,321).  visible(mc1gt_1,321). happensAt(walking(mc1gt_1),321). orientation(mc1gt_2,1,321). coords(mc1gt_2,225,224,321).  visible(mc1gt_2,321). happensAt(walking(mc1gt_2),321).
#show holdsAt/2.