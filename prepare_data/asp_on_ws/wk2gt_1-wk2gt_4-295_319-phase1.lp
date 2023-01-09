
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

	orientation(wk2gt_1,125,295). coords(wk2gt_1,115,53,295).  visible(wk2gt_1,295). happensAt(inactive(wk2gt_1),295). orientation(wk2gt_4,40,295). coords(wk2gt_4,274,285,295).  happensAt(appear(wk2gt_4),295). happensAt(walking(wk2gt_4),295).
orientation(wk2gt_1,125,296). coords(wk2gt_1,115,53,296).  visible(wk2gt_1,296). happensAt(inactive(wk2gt_1),296). orientation(wk2gt_4,61,296). coords(wk2gt_4,272,280,296).  visible(wk2gt_4,296). happensAt(walking(wk2gt_4),296).
orientation(wk2gt_1,125,297). coords(wk2gt_1,115,53,297).  visible(wk2gt_1,297). happensAt(inactive(wk2gt_1),297). orientation(wk2gt_4,61,297). coords(wk2gt_4,264,277,297).  visible(wk2gt_4,297). happensAt(walking(wk2gt_4),297).
orientation(wk2gt_1,125,298). coords(wk2gt_1,115,53,298).  visible(wk2gt_1,298). happensAt(inactive(wk2gt_1),298). orientation(wk2gt_4,61,298). coords(wk2gt_4,262,274,298).  visible(wk2gt_4,298). happensAt(walking(wk2gt_4),298).
orientation(wk2gt_1,125,299). coords(wk2gt_1,115,53,299).  visible(wk2gt_1,299). happensAt(inactive(wk2gt_1),299). orientation(wk2gt_4,61,299). coords(wk2gt_4,261,273,299).  visible(wk2gt_4,299). happensAt(walking(wk2gt_4),299).
orientation(wk2gt_1,125,300). coords(wk2gt_1,115,53,300).  visible(wk2gt_1,300). happensAt(inactive(wk2gt_1),300). orientation(wk2gt_4,61,300). coords(wk2gt_4,264,271,300).  visible(wk2gt_4,300). happensAt(walking(wk2gt_4),300).
orientation(wk2gt_1,125,301). coords(wk2gt_1,115,53,301).  visible(wk2gt_1,301). happensAt(inactive(wk2gt_1),301). orientation(wk2gt_4,61,301). coords(wk2gt_4,260,271,301).  visible(wk2gt_4,301). happensAt(walking(wk2gt_4),301).
orientation(wk2gt_1,125,302). coords(wk2gt_1,115,53,302).  visible(wk2gt_1,302). happensAt(inactive(wk2gt_1),302). orientation(wk2gt_4,53,302). coords(wk2gt_4,259,271,302).  visible(wk2gt_4,302). happensAt(walking(wk2gt_4),302).
orientation(wk2gt_1,125,303). coords(wk2gt_1,115,53,303).  visible(wk2gt_1,303). happensAt(inactive(wk2gt_1),303). orientation(wk2gt_4,52,303). coords(wk2gt_4,259,271,303).  visible(wk2gt_4,303). happensAt(walking(wk2gt_4),303).
orientation(wk2gt_1,125,304). coords(wk2gt_1,115,53,304).  visible(wk2gt_1,304). happensAt(inactive(wk2gt_1),304). orientation(wk2gt_4,48,304). coords(wk2gt_4,258,271,304).  visible(wk2gt_4,304). happensAt(walking(wk2gt_4),304).
orientation(wk2gt_1,125,305). coords(wk2gt_1,115,53,305).  visible(wk2gt_1,305). happensAt(inactive(wk2gt_1),305). orientation(wk2gt_4,47,305). coords(wk2gt_4,258,270,305).  visible(wk2gt_4,305). happensAt(walking(wk2gt_4),305).
orientation(wk2gt_1,125,306). coords(wk2gt_1,115,53,306).  visible(wk2gt_1,306). happensAt(inactive(wk2gt_1),306). orientation(wk2gt_4,47,306). coords(wk2gt_4,257,268,306).  visible(wk2gt_4,306). happensAt(walking(wk2gt_4),306).
orientation(wk2gt_1,125,307). coords(wk2gt_1,115,53,307).  visible(wk2gt_1,307). happensAt(inactive(wk2gt_1),307). orientation(wk2gt_4,47,307). coords(wk2gt_4,257,266,307).  visible(wk2gt_4,307). happensAt(walking(wk2gt_4),307).
orientation(wk2gt_1,125,308). coords(wk2gt_1,115,53,308).  visible(wk2gt_1,308). happensAt(inactive(wk2gt_1),308). orientation(wk2gt_4,36,308). coords(wk2gt_4,256,264,308).  visible(wk2gt_4,308). happensAt(walking(wk2gt_4),308).
orientation(wk2gt_1,125,309). coords(wk2gt_1,115,53,309).  visible(wk2gt_1,309). happensAt(inactive(wk2gt_1),309). orientation(wk2gt_4,36,309). coords(wk2gt_4,255,259,309).  visible(wk2gt_4,309). happensAt(walking(wk2gt_4),309).
orientation(wk2gt_1,125,310). coords(wk2gt_1,115,53,310).  visible(wk2gt_1,310). happensAt(inactive(wk2gt_1),310). orientation(wk2gt_4,36,310). coords(wk2gt_4,255,255,310).  visible(wk2gt_4,310). happensAt(walking(wk2gt_4),310).
orientation(wk2gt_1,125,311). coords(wk2gt_1,115,53,311).  visible(wk2gt_1,311). happensAt(inactive(wk2gt_1),311). orientation(wk2gt_4,36,311). coords(wk2gt_4,254,252,311).  visible(wk2gt_4,311). happensAt(walking(wk2gt_4),311).
orientation(wk2gt_1,125,312). coords(wk2gt_1,115,53,312).  visible(wk2gt_1,312). happensAt(inactive(wk2gt_1),312). orientation(wk2gt_4,36,312). coords(wk2gt_4,254,249,312).  visible(wk2gt_4,312). happensAt(walking(wk2gt_4),312).
orientation(wk2gt_1,125,313). coords(wk2gt_1,115,53,313).  visible(wk2gt_1,313). happensAt(inactive(wk2gt_1),313). orientation(wk2gt_4,36,313). coords(wk2gt_4,253,249,313).  visible(wk2gt_4,313). happensAt(walking(wk2gt_4),313).
orientation(wk2gt_1,125,314). coords(wk2gt_1,115,53,314).  visible(wk2gt_1,314). happensAt(inactive(wk2gt_1),314). orientation(wk2gt_4,23,314). coords(wk2gt_4,252,247,314).  visible(wk2gt_4,314). happensAt(walking(wk2gt_4),314).
orientation(wk2gt_1,125,315). coords(wk2gt_1,115,53,315).  visible(wk2gt_1,315). happensAt(inactive(wk2gt_1),315). orientation(wk2gt_4,9,315). coords(wk2gt_4,251,246,315).  visible(wk2gt_4,315). happensAt(walking(wk2gt_4),315).
orientation(wk2gt_1,125,316). coords(wk2gt_1,115,53,316).  visible(wk2gt_1,316). happensAt(inactive(wk2gt_1),316). orientation(wk2gt_4,2,316). coords(wk2gt_4,250,244,316).  visible(wk2gt_4,316). happensAt(walking(wk2gt_4),316).
orientation(wk2gt_1,125,317). coords(wk2gt_1,115,53,317).  visible(wk2gt_1,317). happensAt(inactive(wk2gt_1),317). orientation(wk2gt_4,2,317). coords(wk2gt_4,249,242,317).  visible(wk2gt_4,317). happensAt(walking(wk2gt_4),317).
orientation(wk2gt_1,125,318). coords(wk2gt_1,115,53,318).  visible(wk2gt_1,318). happensAt(inactive(wk2gt_1),318). orientation(wk2gt_4,2,318). coords(wk2gt_4,245,240,318).  visible(wk2gt_4,318). happensAt(walking(wk2gt_4),318).
orientation(wk2gt_1,125,319). coords(wk2gt_1,115,53,319).  visible(wk2gt_1,319). happensAt(inactive(wk2gt_1),319). orientation(wk2gt_4,2,319). coords(wk2gt_4,243,237,319).  visible(wk2gt_4,319). happensAt(walking(wk2gt_4),319).
#show holdsAt/2.