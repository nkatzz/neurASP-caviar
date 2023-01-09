
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

	orientation(wk1gt_1,163,296). coords(wk1gt_1,193,198,296).  visible(wk1gt_1,296). happensAt(walking(wk1gt_1),296). orientation(wk1gt_4,161,296). coords(wk1gt_4,52,176,296).  visible(wk1gt_4,296). happensAt(active(wk1gt_4),296).
orientation(wk1gt_1,163,297). coords(wk1gt_1,191,193,297).  visible(wk1gt_1,297). happensAt(walking(wk1gt_1),297). orientation(wk1gt_4,160,297). coords(wk1gt_4,52,176,297).  visible(wk1gt_4,297). happensAt(active(wk1gt_4),297).
orientation(wk1gt_1,163,298). coords(wk1gt_1,191,192,298).  visible(wk1gt_1,298). happensAt(walking(wk1gt_1),298). orientation(wk1gt_4,161,298). coords(wk1gt_4,52,175,298).  visible(wk1gt_4,298). happensAt(active(wk1gt_4),298).
orientation(wk1gt_1,163,299). coords(wk1gt_1,189,191,299).  visible(wk1gt_1,299). happensAt(walking(wk1gt_1),299). orientation(wk1gt_4,167,299). coords(wk1gt_4,52,175,299).  visible(wk1gt_4,299). happensAt(active(wk1gt_4),299).
orientation(wk1gt_1,155,300). coords(wk1gt_1,189,191,300).  visible(wk1gt_1,300). happensAt(walking(wk1gt_1),300). orientation(wk1gt_4,168,300). coords(wk1gt_4,52,174,300).  visible(wk1gt_4,300). happensAt(active(wk1gt_4),300).
orientation(wk1gt_1,155,301). coords(wk1gt_1,188,190,301).  visible(wk1gt_1,301). happensAt(walking(wk1gt_1),301). orientation(wk1gt_4,173,301). coords(wk1gt_4,52,174,301).  visible(wk1gt_4,301). happensAt(active(wk1gt_4),301).
orientation(wk1gt_1,155,302). coords(wk1gt_1,187,189,302).  visible(wk1gt_1,302). happensAt(walking(wk1gt_1),302). orientation(wk1gt_4,170,302). coords(wk1gt_4,52,173,302).  visible(wk1gt_4,302). happensAt(active(wk1gt_4),302).
orientation(wk1gt_1,155,303). coords(wk1gt_1,187,188,303).  visible(wk1gt_1,303). happensAt(walking(wk1gt_1),303). orientation(wk1gt_4,170,303). coords(wk1gt_4,52,173,303).  visible(wk1gt_4,303). happensAt(active(wk1gt_4),303).
orientation(wk1gt_1,155,304). coords(wk1gt_1,186,187,304).  visible(wk1gt_1,304). happensAt(walking(wk1gt_1),304). orientation(wk1gt_4,170,304). coords(wk1gt_4,52,173,304).  visible(wk1gt_4,304). happensAt(active(wk1gt_4),304).
orientation(wk1gt_1,155,305). coords(wk1gt_1,185,186,305).  visible(wk1gt_1,305). happensAt(walking(wk1gt_1),305). orientation(wk1gt_4,169,305). coords(wk1gt_4,52,173,305).  visible(wk1gt_4,305). happensAt(active(wk1gt_4),305).
orientation(wk1gt_1,155,306). coords(wk1gt_1,183,184,306).  visible(wk1gt_1,306). happensAt(walking(wk1gt_1),306). orientation(wk1gt_4,168,306). coords(wk1gt_4,52,173,306).  visible(wk1gt_4,306). happensAt(active(wk1gt_4),306).
orientation(wk1gt_1,155,307). coords(wk1gt_1,182,181,307).  visible(wk1gt_1,307). happensAt(walking(wk1gt_1),307). orientation(wk1gt_4,167,307). coords(wk1gt_4,52,173,307).  visible(wk1gt_4,307). happensAt(active(wk1gt_4),307).
orientation(wk1gt_1,142,308). coords(wk1gt_1,179,179,308).  visible(wk1gt_1,308). happensAt(walking(wk1gt_1),308). orientation(wk1gt_4,167,308). coords(wk1gt_4,52,173,308).  visible(wk1gt_4,308). happensAt(active(wk1gt_4),308).
orientation(wk1gt_1,151,309). coords(wk1gt_1,178,175,309).  visible(wk1gt_1,309). happensAt(walking(wk1gt_1),309). orientation(wk1gt_4,168,309). coords(wk1gt_4,52,174,309).  visible(wk1gt_4,309). happensAt(active(wk1gt_4),309).
orientation(wk1gt_1,151,310). coords(wk1gt_1,177,173,310).  visible(wk1gt_1,310). happensAt(walking(wk1gt_1),310). orientation(wk1gt_4,164,310). coords(wk1gt_4,52,174,310).  visible(wk1gt_4,310). happensAt(active(wk1gt_4),310).
orientation(wk1gt_1,151,311). coords(wk1gt_1,177,170,311).  visible(wk1gt_1,311). happensAt(walking(wk1gt_1),311). orientation(wk1gt_4,162,311). coords(wk1gt_4,52,175,311).  visible(wk1gt_4,311). happensAt(active(wk1gt_4),311).
orientation(wk1gt_1,151,312). coords(wk1gt_1,176,170,312).  visible(wk1gt_1,312). happensAt(walking(wk1gt_1),312). orientation(wk1gt_4,161,312). coords(wk1gt_4,52,175,312).  visible(wk1gt_4,312). happensAt(active(wk1gt_4),312).
orientation(wk1gt_1,151,313). coords(wk1gt_1,176,169,313).  visible(wk1gt_1,313). happensAt(walking(wk1gt_1),313). orientation(wk1gt_4,162,313). coords(wk1gt_4,52,175,313).  visible(wk1gt_4,313). happensAt(active(wk1gt_4),313).
orientation(wk1gt_1,151,314). coords(wk1gt_1,176,169,314).  visible(wk1gt_1,314). happensAt(walking(wk1gt_1),314). orientation(wk1gt_4,161,314). coords(wk1gt_4,52,176,314).  visible(wk1gt_4,314). happensAt(active(wk1gt_4),314).
orientation(wk1gt_1,151,315). coords(wk1gt_1,175,168,315).  visible(wk1gt_1,315). happensAt(walking(wk1gt_1),315). orientation(wk1gt_4,159,315). coords(wk1gt_4,52,176,315).  visible(wk1gt_4,315). happensAt(inactive(wk1gt_4),315).
orientation(wk1gt_1,151,316). coords(wk1gt_1,175,167,316).  visible(wk1gt_1,316). happensAt(walking(wk1gt_1),316). orientation(wk1gt_4,157,316). coords(wk1gt_4,52,176,316).  visible(wk1gt_4,316). happensAt(inactive(wk1gt_4),316).
orientation(wk1gt_1,151,317). coords(wk1gt_1,174,166,317).  visible(wk1gt_1,317). happensAt(walking(wk1gt_1),317). orientation(wk1gt_4,160,317). coords(wk1gt_4,52,176,317).  visible(wk1gt_4,317). happensAt(inactive(wk1gt_4),317).
orientation(wk1gt_1,151,318). coords(wk1gt_1,173,165,318).  visible(wk1gt_1,318). happensAt(walking(wk1gt_1),318). orientation(wk1gt_4,160,318). coords(wk1gt_4,52,176,318).  visible(wk1gt_4,318). happensAt(inactive(wk1gt_4),318).
orientation(wk1gt_1,151,319). coords(wk1gt_1,171,163,319).  visible(wk1gt_1,319). happensAt(walking(wk1gt_1),319). orientation(wk1gt_4,160,319). coords(wk1gt_4,52,176,319).  visible(wk1gt_4,319). happensAt(inactive(wk1gt_4),319).
orientation(wk1gt_1,151,320). coords(wk1gt_1,170,161,320).  visible(wk1gt_1,320). happensAt(walking(wk1gt_1),320). orientation(wk1gt_4,160,320). coords(wk1gt_4,52,176,320).  visible(wk1gt_4,320). happensAt(inactive(wk1gt_4),320).
#show holdsAt/2.