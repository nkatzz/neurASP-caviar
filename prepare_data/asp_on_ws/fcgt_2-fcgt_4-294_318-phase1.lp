
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

	orientation(fcgt_2,150,294). coords(fcgt_2,86,132,294).  visible(fcgt_2,294). happensAt(walking(fcgt_2),294). orientation(fcgt_4,166,294). coords(fcgt_4,82,206,294).  visible(fcgt_4,294). happensAt(inactive(fcgt_4),294).
orientation(fcgt_2,150,295). coords(fcgt_2,86,132,295).  visible(fcgt_2,295). happensAt(walking(fcgt_2),295). orientation(fcgt_4,166,295). coords(fcgt_4,82,206,295).  visible(fcgt_4,295). happensAt(inactive(fcgt_4),295).
orientation(fcgt_2,150,296). coords(fcgt_2,86,131,296).  visible(fcgt_2,296). happensAt(walking(fcgt_2),296). orientation(fcgt_4,166,296). coords(fcgt_4,82,206,296).  visible(fcgt_4,296). happensAt(inactive(fcgt_4),296).
orientation(fcgt_2,150,297). coords(fcgt_2,86,131,297).  visible(fcgt_2,297). happensAt(active(fcgt_2),297). orientation(fcgt_4,166,297). coords(fcgt_4,82,206,297).  visible(fcgt_4,297). happensAt(inactive(fcgt_4),297).
orientation(fcgt_2,150,298). coords(fcgt_2,87,131,298).  visible(fcgt_2,298). happensAt(active(fcgt_2),298). orientation(fcgt_4,166,298). coords(fcgt_4,82,206,298).  visible(fcgt_4,298). happensAt(inactive(fcgt_4),298).
orientation(fcgt_2,150,299). coords(fcgt_2,88,131,299).  visible(fcgt_2,299). happensAt(active(fcgt_2),299). orientation(fcgt_4,166,299). coords(fcgt_4,82,206,299).  visible(fcgt_4,299). happensAt(inactive(fcgt_4),299).
orientation(fcgt_2,150,300). coords(fcgt_2,90,130,300).  visible(fcgt_2,300). happensAt(active(fcgt_2),300). orientation(fcgt_4,166,300). coords(fcgt_4,82,206,300).  visible(fcgt_4,300). happensAt(inactive(fcgt_4),300).
orientation(fcgt_2,150,301). coords(fcgt_2,91,130,301).  visible(fcgt_2,301). happensAt(active(fcgt_2),301). orientation(fcgt_4,166,301). coords(fcgt_4,82,206,301).  visible(fcgt_4,301). happensAt(inactive(fcgt_4),301).
orientation(fcgt_2,150,302). coords(fcgt_2,91,130,302).  visible(fcgt_2,302). happensAt(active(fcgt_2),302). orientation(fcgt_4,166,302). coords(fcgt_4,82,206,302).  visible(fcgt_4,302). happensAt(inactive(fcgt_4),302).
orientation(fcgt_2,150,303). coords(fcgt_2,92,130,303).  visible(fcgt_2,303). happensAt(active(fcgt_2),303). orientation(fcgt_4,166,303). coords(fcgt_4,82,206,303).  visible(fcgt_4,303). happensAt(inactive(fcgt_4),303).
orientation(fcgt_2,150,304). coords(fcgt_2,92,130,304).  visible(fcgt_2,304). happensAt(active(fcgt_2),304). orientation(fcgt_4,166,304). coords(fcgt_4,82,206,304).  visible(fcgt_4,304). happensAt(inactive(fcgt_4),304).
orientation(fcgt_2,150,305). coords(fcgt_2,92,130,305).  visible(fcgt_2,305). happensAt(active(fcgt_2),305). orientation(fcgt_4,166,305). coords(fcgt_4,82,206,305).  visible(fcgt_4,305). happensAt(inactive(fcgt_4),305).
orientation(fcgt_2,150,306). coords(fcgt_2,92,129,306).  visible(fcgt_2,306). happensAt(active(fcgt_2),306). orientation(fcgt_4,166,306). coords(fcgt_4,82,206,306).  visible(fcgt_4,306). happensAt(inactive(fcgt_4),306).
orientation(fcgt_2,150,307). coords(fcgt_2,92,129,307).  visible(fcgt_2,307). happensAt(active(fcgt_2),307). orientation(fcgt_4,166,307). coords(fcgt_4,82,206,307).  visible(fcgt_4,307). happensAt(inactive(fcgt_4),307).
orientation(fcgt_2,150,308). coords(fcgt_2,92,129,308).  visible(fcgt_2,308). happensAt(active(fcgt_2),308). orientation(fcgt_4,166,308). coords(fcgt_4,82,206,308).  visible(fcgt_4,308). happensAt(inactive(fcgt_4),308).
orientation(fcgt_2,150,309). coords(fcgt_2,92,129,309).  visible(fcgt_2,309). happensAt(active(fcgt_2),309). orientation(fcgt_4,166,309). coords(fcgt_4,82,206,309).  visible(fcgt_4,309). happensAt(inactive(fcgt_4),309).
orientation(fcgt_2,150,310). coords(fcgt_2,93,129,310).  visible(fcgt_2,310). happensAt(active(fcgt_2),310). orientation(fcgt_4,166,310). coords(fcgt_4,82,206,310).  visible(fcgt_4,310). happensAt(inactive(fcgt_4),310).
orientation(fcgt_2,150,311). coords(fcgt_2,94,129,311).  visible(fcgt_2,311). happensAt(active(fcgt_2),311). orientation(fcgt_4,166,311). coords(fcgt_4,82,206,311).  visible(fcgt_4,311). happensAt(inactive(fcgt_4),311).
orientation(fcgt_2,150,312). coords(fcgt_2,93,129,312).  visible(fcgt_2,312). happensAt(active(fcgt_2),312). orientation(fcgt_4,166,312). coords(fcgt_4,82,206,312).  visible(fcgt_4,312). happensAt(inactive(fcgt_4),312).
orientation(fcgt_2,150,313). coords(fcgt_2,95,129,313).  visible(fcgt_2,313). happensAt(active(fcgt_2),313). orientation(fcgt_4,166,313). coords(fcgt_4,82,206,313).  visible(fcgt_4,313). happensAt(inactive(fcgt_4),313).
orientation(fcgt_2,150,314). coords(fcgt_2,95,129,314).  visible(fcgt_2,314). happensAt(active(fcgt_2),314). orientation(fcgt_4,166,314). coords(fcgt_4,82,206,314).  visible(fcgt_4,314). happensAt(inactive(fcgt_4),314).
orientation(fcgt_2,150,315). coords(fcgt_2,95,129,315).  visible(fcgt_2,315). happensAt(active(fcgt_2),315). orientation(fcgt_4,166,315). coords(fcgt_4,82,206,315).  visible(fcgt_4,315). happensAt(inactive(fcgt_4),315).
orientation(fcgt_2,150,316). coords(fcgt_2,95,129,316).  visible(fcgt_2,316). happensAt(active(fcgt_2),316). orientation(fcgt_4,166,316). coords(fcgt_4,82,206,316).  visible(fcgt_4,316). happensAt(inactive(fcgt_4),316).
orientation(fcgt_2,150,317). coords(fcgt_2,95,129,317).  visible(fcgt_2,317). happensAt(active(fcgt_2),317). orientation(fcgt_4,166,317). coords(fcgt_4,82,206,317).  visible(fcgt_4,317). happensAt(inactive(fcgt_4),317).
orientation(fcgt_2,150,318). coords(fcgt_2,95,129,318).  visible(fcgt_2,318). happensAt(active(fcgt_2),318). orientation(fcgt_4,166,318). coords(fcgt_4,82,206,318).  visible(fcgt_4,318). happensAt(inactive(fcgt_4),318).
#show holdsAt/2.