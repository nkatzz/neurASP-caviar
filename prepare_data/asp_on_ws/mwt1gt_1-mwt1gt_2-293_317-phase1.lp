
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

	orientation(mwt1gt_1,160,293). coords(mwt1gt_1,97,167,293).  visible(mwt1gt_1,293). happensAt(walking(mwt1gt_1),293). orientation(mwt1gt_2,130,293). coords(mwt1gt_2,172,86,293).  visible(mwt1gt_2,293). happensAt(walking(mwt1gt_2),293).
orientation(mwt1gt_1,160,294). coords(mwt1gt_1,97,167,294).  visible(mwt1gt_1,294). happensAt(walking(mwt1gt_1),294). orientation(mwt1gt_2,130,294). coords(mwt1gt_2,171,87,294).  visible(mwt1gt_2,294). happensAt(walking(mwt1gt_2),294).
orientation(mwt1gt_1,160,295). coords(mwt1gt_1,98,167,295).  visible(mwt1gt_1,295). happensAt(walking(mwt1gt_1),295). orientation(mwt1gt_2,130,295). coords(mwt1gt_2,171,89,295).  visible(mwt1gt_2,295). happensAt(walking(mwt1gt_2),295).
orientation(mwt1gt_1,160,296). coords(mwt1gt_1,99,167,296).  visible(mwt1gt_1,296). happensAt(walking(mwt1gt_1),296). orientation(mwt1gt_2,130,296). coords(mwt1gt_2,168,89,296).  visible(mwt1gt_2,296). happensAt(walking(mwt1gt_2),296).
orientation(mwt1gt_1,160,297). coords(mwt1gt_1,99,167,297).  visible(mwt1gt_1,297). happensAt(walking(mwt1gt_1),297). orientation(mwt1gt_2,130,297). coords(mwt1gt_2,166,90,297).  visible(mwt1gt_2,297). happensAt(walking(mwt1gt_2),297).
orientation(mwt1gt_1,160,298). coords(mwt1gt_1,100,167,298).  visible(mwt1gt_1,298). happensAt(walking(mwt1gt_1),298). orientation(mwt1gt_2,130,298). coords(mwt1gt_2,164,90,298).  visible(mwt1gt_2,298). happensAt(walking(mwt1gt_2),298).
orientation(mwt1gt_1,160,299). coords(mwt1gt_1,100,167,299).  visible(mwt1gt_1,299). happensAt(walking(mwt1gt_1),299). orientation(mwt1gt_2,130,299). coords(mwt1gt_2,162,92,299).  visible(mwt1gt_2,299). happensAt(walking(mwt1gt_2),299).
orientation(mwt1gt_1,160,300). coords(mwt1gt_1,100,167,300).  visible(mwt1gt_1,300). happensAt(walking(mwt1gt_1),300). orientation(mwt1gt_2,140,300). coords(mwt1gt_2,159,93,300).  visible(mwt1gt_2,300). happensAt(walking(mwt1gt_2),300).
orientation(mwt1gt_1,160,301). coords(mwt1gt_1,100,165,301).  visible(mwt1gt_1,301). happensAt(walking(mwt1gt_1),301). orientation(mwt1gt_2,140,301). coords(mwt1gt_2,158,93,301).  visible(mwt1gt_2,301). happensAt(walking(mwt1gt_2),301).
orientation(mwt1gt_1,160,302). coords(mwt1gt_1,101,165,302).  visible(mwt1gt_1,302). happensAt(walking(mwt1gt_1),302). orientation(mwt1gt_2,140,302). coords(mwt1gt_2,157,94,302).  visible(mwt1gt_2,302). happensAt(walking(mwt1gt_2),302).
orientation(mwt1gt_1,160,303). coords(mwt1gt_1,101,165,303).  visible(mwt1gt_1,303). happensAt(walking(mwt1gt_1),303). orientation(mwt1gt_2,140,303). coords(mwt1gt_2,156,96,303).  visible(mwt1gt_2,303). happensAt(walking(mwt1gt_2),303).
orientation(mwt1gt_1,160,304). coords(mwt1gt_1,101,165,304).  visible(mwt1gt_1,304). happensAt(walking(mwt1gt_1),304). orientation(mwt1gt_2,140,304). coords(mwt1gt_2,156,99,304).  visible(mwt1gt_2,304). happensAt(walking(mwt1gt_2),304).
orientation(mwt1gt_1,160,305). coords(mwt1gt_1,103,165,305).  visible(mwt1gt_1,305). happensAt(walking(mwt1gt_1),305). orientation(mwt1gt_2,140,305). coords(mwt1gt_2,155,99,305).  visible(mwt1gt_2,305). happensAt(walking(mwt1gt_2),305).
orientation(mwt1gt_1,160,306). coords(mwt1gt_1,104,165,306).  visible(mwt1gt_1,306). happensAt(walking(mwt1gt_1),306). orientation(mwt1gt_2,140,306). coords(mwt1gt_2,153,102,306).  visible(mwt1gt_2,306). happensAt(walking(mwt1gt_2),306).
orientation(mwt1gt_1,160,307). coords(mwt1gt_1,106,164,307).  visible(mwt1gt_1,307). happensAt(walking(mwt1gt_1),307). orientation(mwt1gt_2,140,307). coords(mwt1gt_2,153,104,307).  visible(mwt1gt_2,307). happensAt(walking(mwt1gt_2),307).
orientation(mwt1gt_1,160,308). coords(mwt1gt_1,107,164,308).  visible(mwt1gt_1,308). happensAt(walking(mwt1gt_1),308). orientation(mwt1gt_2,140,308). coords(mwt1gt_2,152,104,308).  visible(mwt1gt_2,308). happensAt(walking(mwt1gt_2),308).
orientation(mwt1gt_1,160,309). coords(mwt1gt_1,107,163,309).  visible(mwt1gt_1,309). happensAt(walking(mwt1gt_1),309). orientation(mwt1gt_2,140,309). coords(mwt1gt_2,152,106,309).  visible(mwt1gt_2,309). happensAt(walking(mwt1gt_2),309).
orientation(mwt1gt_1,160,310). coords(mwt1gt_1,107,163,310).  visible(mwt1gt_1,310). happensAt(walking(mwt1gt_1),310). orientation(mwt1gt_2,140,310). coords(mwt1gt_2,152,106,310).  visible(mwt1gt_2,310). happensAt(walking(mwt1gt_2),310).
orientation(mwt1gt_1,160,311). coords(mwt1gt_1,107,162,311).  visible(mwt1gt_1,311). happensAt(walking(mwt1gt_1),311). orientation(mwt1gt_2,140,311). coords(mwt1gt_2,145,108,311).  visible(mwt1gt_2,311). happensAt(walking(mwt1gt_2),311).
orientation(mwt1gt_1,160,312). coords(mwt1gt_1,109,160,312).  visible(mwt1gt_1,312). happensAt(walking(mwt1gt_1),312). orientation(mwt1gt_2,140,312). coords(mwt1gt_2,145,108,312).  visible(mwt1gt_2,312). happensAt(walking(mwt1gt_2),312).
orientation(mwt1gt_1,160,313). coords(mwt1gt_1,109,160,313).  visible(mwt1gt_1,313). happensAt(walking(mwt1gt_1),313). orientation(mwt1gt_2,140,313). coords(mwt1gt_2,143,108,313).  visible(mwt1gt_2,313). happensAt(walking(mwt1gt_2),313).
orientation(mwt1gt_1,160,314). coords(mwt1gt_1,109,160,314).  visible(mwt1gt_1,314). happensAt(walking(mwt1gt_1),314). orientation(mwt1gt_2,140,314). coords(mwt1gt_2,141,109,314).  visible(mwt1gt_2,314). happensAt(walking(mwt1gt_2),314).
orientation(mwt1gt_1,160,315). coords(mwt1gt_1,109,158,315).  visible(mwt1gt_1,315). happensAt(walking(mwt1gt_1),315). orientation(mwt1gt_2,140,315). coords(mwt1gt_2,139,112,315).  visible(mwt1gt_2,315). happensAt(walking(mwt1gt_2),315).
orientation(mwt1gt_1,160,316). coords(mwt1gt_1,109,157,316).  visible(mwt1gt_1,316). happensAt(walking(mwt1gt_1),316). orientation(mwt1gt_2,140,316). coords(mwt1gt_2,138,115,316).  visible(mwt1gt_2,316). happensAt(walking(mwt1gt_2),316).
orientation(mwt1gt_1,160,317). coords(mwt1gt_1,111,155,317).  visible(mwt1gt_1,317). happensAt(walking(mwt1gt_1),317). orientation(mwt1gt_2,140,317). coords(mwt1gt_2,137,116,317).  visible(mwt1gt_2,317). happensAt(walking(mwt1gt_2),317).
#show holdsAt/2.