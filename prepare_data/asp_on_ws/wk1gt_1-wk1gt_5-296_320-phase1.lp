
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

	orientation(wk1gt_1,163,296). coords(wk1gt_1,193,198,296).  visible(wk1gt_1,296). happensAt(walking(wk1gt_1),296). orientation(wk1gt_5,162,296). coords(wk1gt_5,50,164,296).  visible(wk1gt_5,296). happensAt(active(wk1gt_5),296).
orientation(wk1gt_1,163,297). coords(wk1gt_1,191,193,297).  visible(wk1gt_1,297). happensAt(walking(wk1gt_1),297). orientation(wk1gt_5,162,297). coords(wk1gt_5,50,164,297).  visible(wk1gt_5,297). happensAt(active(wk1gt_5),297).
orientation(wk1gt_1,163,298). coords(wk1gt_1,191,192,298).  visible(wk1gt_1,298). happensAt(walking(wk1gt_1),298). orientation(wk1gt_5,162,298). coords(wk1gt_5,50,164,298).  visible(wk1gt_5,298). happensAt(active(wk1gt_5),298).
orientation(wk1gt_1,163,299). coords(wk1gt_1,189,191,299).  visible(wk1gt_1,299). happensAt(walking(wk1gt_1),299). orientation(wk1gt_5,162,299). coords(wk1gt_5,50,164,299).  visible(wk1gt_5,299). happensAt(active(wk1gt_5),299).
orientation(wk1gt_1,155,300). coords(wk1gt_1,189,191,300).  visible(wk1gt_1,300). happensAt(walking(wk1gt_1),300). orientation(wk1gt_5,162,300). coords(wk1gt_5,50,164,300).  visible(wk1gt_5,300). happensAt(active(wk1gt_5),300).
orientation(wk1gt_1,155,301). coords(wk1gt_1,188,190,301).  visible(wk1gt_1,301). happensAt(walking(wk1gt_1),301). orientation(wk1gt_5,162,301). coords(wk1gt_5,50,164,301).  visible(wk1gt_5,301). happensAt(active(wk1gt_5),301).
orientation(wk1gt_1,155,302). coords(wk1gt_1,187,189,302).  visible(wk1gt_1,302). happensAt(walking(wk1gt_1),302). orientation(wk1gt_5,162,302). coords(wk1gt_5,51,164,302).  visible(wk1gt_5,302). happensAt(active(wk1gt_5),302).
orientation(wk1gt_1,155,303). coords(wk1gt_1,187,188,303).  visible(wk1gt_1,303). happensAt(walking(wk1gt_1),303). orientation(wk1gt_5,162,303). coords(wk1gt_5,50,164,303).  visible(wk1gt_5,303). happensAt(active(wk1gt_5),303).
orientation(wk1gt_1,155,304). coords(wk1gt_1,186,187,304).  visible(wk1gt_1,304). happensAt(walking(wk1gt_1),304). orientation(wk1gt_5,162,304). coords(wk1gt_5,50,163,304).  visible(wk1gt_5,304). happensAt(active(wk1gt_5),304).
orientation(wk1gt_1,155,305). coords(wk1gt_1,185,186,305).  visible(wk1gt_1,305). happensAt(walking(wk1gt_1),305). orientation(wk1gt_5,162,305). coords(wk1gt_5,51,163,305).  visible(wk1gt_5,305). happensAt(active(wk1gt_5),305).
orientation(wk1gt_1,155,306). coords(wk1gt_1,183,184,306).  visible(wk1gt_1,306). happensAt(walking(wk1gt_1),306). orientation(wk1gt_5,162,306). coords(wk1gt_5,51,163,306).  visible(wk1gt_5,306). happensAt(active(wk1gt_5),306).
orientation(wk1gt_1,155,307). coords(wk1gt_1,182,181,307).  visible(wk1gt_1,307). happensAt(walking(wk1gt_1),307). orientation(wk1gt_5,162,307). coords(wk1gt_5,51,163,307).  visible(wk1gt_5,307). happensAt(active(wk1gt_5),307).
orientation(wk1gt_1,142,308). coords(wk1gt_1,179,179,308).  visible(wk1gt_1,308). happensAt(walking(wk1gt_1),308). orientation(wk1gt_5,162,308). coords(wk1gt_5,50,163,308).  visible(wk1gt_5,308). happensAt(active(wk1gt_5),308).
orientation(wk1gt_1,151,309). coords(wk1gt_1,178,175,309).  visible(wk1gt_1,309). happensAt(walking(wk1gt_1),309). orientation(wk1gt_5,162,309). coords(wk1gt_5,50,163,309).  visible(wk1gt_5,309). happensAt(active(wk1gt_5),309).
orientation(wk1gt_1,151,310). coords(wk1gt_1,177,173,310).  visible(wk1gt_1,310). happensAt(walking(wk1gt_1),310). orientation(wk1gt_5,162,310). coords(wk1gt_5,50,163,310).  visible(wk1gt_5,310). happensAt(active(wk1gt_5),310).
orientation(wk1gt_1,151,311). coords(wk1gt_1,177,170,311).  visible(wk1gt_1,311). happensAt(walking(wk1gt_1),311). orientation(wk1gt_5,162,311). coords(wk1gt_5,50,163,311).  visible(wk1gt_5,311). happensAt(active(wk1gt_5),311).
orientation(wk1gt_1,151,312). coords(wk1gt_1,176,170,312).  visible(wk1gt_1,312). happensAt(walking(wk1gt_1),312). orientation(wk1gt_5,162,312). coords(wk1gt_5,50,163,312).  visible(wk1gt_5,312). happensAt(inactive(wk1gt_5),312).
orientation(wk1gt_1,151,313). coords(wk1gt_1,176,169,313).  visible(wk1gt_1,313). happensAt(walking(wk1gt_1),313). orientation(wk1gt_5,162,313). coords(wk1gt_5,50,163,313).  visible(wk1gt_5,313). happensAt(inactive(wk1gt_5),313).
orientation(wk1gt_1,151,314). coords(wk1gt_1,176,169,314).  visible(wk1gt_1,314). happensAt(walking(wk1gt_1),314). orientation(wk1gt_5,162,314). coords(wk1gt_5,51,163,314).  visible(wk1gt_5,314). happensAt(inactive(wk1gt_5),314).
orientation(wk1gt_1,151,315). coords(wk1gt_1,175,168,315).  visible(wk1gt_1,315). happensAt(walking(wk1gt_1),315). orientation(wk1gt_5,162,315). coords(wk1gt_5,50,163,315).  visible(wk1gt_5,315). happensAt(inactive(wk1gt_5),315).
orientation(wk1gt_1,151,316). coords(wk1gt_1,175,167,316).  visible(wk1gt_1,316). happensAt(walking(wk1gt_1),316). orientation(wk1gt_5,162,316). coords(wk1gt_5,50,163,316).  visible(wk1gt_5,316). happensAt(inactive(wk1gt_5),316).
orientation(wk1gt_1,151,317). coords(wk1gt_1,174,166,317).  visible(wk1gt_1,317). happensAt(walking(wk1gt_1),317). orientation(wk1gt_5,162,317). coords(wk1gt_5,50,163,317).  visible(wk1gt_5,317). happensAt(inactive(wk1gt_5),317).
orientation(wk1gt_1,151,318). coords(wk1gt_1,173,165,318).  visible(wk1gt_1,318). happensAt(walking(wk1gt_1),318). orientation(wk1gt_5,162,318). coords(wk1gt_5,50,163,318).  visible(wk1gt_5,318). happensAt(inactive(wk1gt_5),318).
orientation(wk1gt_1,151,319). coords(wk1gt_1,171,163,319).  visible(wk1gt_1,319). happensAt(walking(wk1gt_1),319). orientation(wk1gt_5,162,319). coords(wk1gt_5,50,163,319).  visible(wk1gt_5,319). happensAt(inactive(wk1gt_5),319).
orientation(wk1gt_1,151,320). coords(wk1gt_1,170,161,320).  visible(wk1gt_1,320). happensAt(walking(wk1gt_1),320). orientation(wk1gt_5,162,320). coords(wk1gt_5,51,163,320).  visible(wk1gt_5,320). happensAt(inactive(wk1gt_5),320).
#show holdsAt/2.