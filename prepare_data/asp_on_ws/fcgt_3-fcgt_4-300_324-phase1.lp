
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

	orientation(fcgt_3,164,300). coords(fcgt_3,98,187,300).  visible(fcgt_3,300). happensAt(inactive(fcgt_3),300). orientation(fcgt_4,166,300). coords(fcgt_4,82,206,300).  visible(fcgt_4,300). happensAt(inactive(fcgt_4),300).
orientation(fcgt_3,164,301). coords(fcgt_3,98,187,301).  visible(fcgt_3,301). happensAt(inactive(fcgt_3),301). orientation(fcgt_4,166,301). coords(fcgt_4,82,206,301).  visible(fcgt_4,301). happensAt(inactive(fcgt_4),301).
orientation(fcgt_3,164,302). coords(fcgt_3,98,187,302).  visible(fcgt_3,302). happensAt(inactive(fcgt_3),302). orientation(fcgt_4,166,302). coords(fcgt_4,82,206,302).  visible(fcgt_4,302). happensAt(inactive(fcgt_4),302).
orientation(fcgt_3,164,303). coords(fcgt_3,98,187,303).  visible(fcgt_3,303). happensAt(inactive(fcgt_3),303). orientation(fcgt_4,166,303). coords(fcgt_4,82,206,303).  visible(fcgt_4,303). happensAt(inactive(fcgt_4),303).
orientation(fcgt_3,164,304). coords(fcgt_3,98,187,304).  visible(fcgt_3,304). happensAt(inactive(fcgt_3),304). orientation(fcgt_4,166,304). coords(fcgt_4,82,206,304).  visible(fcgt_4,304). happensAt(inactive(fcgt_4),304).
orientation(fcgt_3,164,305). coords(fcgt_3,98,187,305).  visible(fcgt_3,305). happensAt(inactive(fcgt_3),305). orientation(fcgt_4,166,305). coords(fcgt_4,82,206,305).  visible(fcgt_4,305). happensAt(inactive(fcgt_4),305).
orientation(fcgt_3,164,306). coords(fcgt_3,98,187,306).  visible(fcgt_3,306). happensAt(inactive(fcgt_3),306). orientation(fcgt_4,166,306). coords(fcgt_4,82,206,306).  visible(fcgt_4,306). happensAt(inactive(fcgt_4),306).
orientation(fcgt_3,164,307). coords(fcgt_3,98,187,307).  visible(fcgt_3,307). happensAt(inactive(fcgt_3),307). orientation(fcgt_4,166,307). coords(fcgt_4,82,206,307).  visible(fcgt_4,307). happensAt(inactive(fcgt_4),307).
orientation(fcgt_3,164,308). coords(fcgt_3,98,187,308).  visible(fcgt_3,308). happensAt(inactive(fcgt_3),308). orientation(fcgt_4,166,308). coords(fcgt_4,82,206,308).  visible(fcgt_4,308). happensAt(inactive(fcgt_4),308).
orientation(fcgt_3,164,309). coords(fcgt_3,98,187,309).  visible(fcgt_3,309). happensAt(inactive(fcgt_3),309). orientation(fcgt_4,166,309). coords(fcgt_4,82,206,309).  visible(fcgt_4,309). happensAt(inactive(fcgt_4),309).
orientation(fcgt_3,164,310). coords(fcgt_3,98,187,310).  visible(fcgt_3,310). happensAt(inactive(fcgt_3),310). orientation(fcgt_4,166,310). coords(fcgt_4,82,206,310).  visible(fcgt_4,310). happensAt(inactive(fcgt_4),310).
orientation(fcgt_3,164,311). coords(fcgt_3,98,187,311).  visible(fcgt_3,311). happensAt(inactive(fcgt_3),311). orientation(fcgt_4,166,311). coords(fcgt_4,82,206,311).  visible(fcgt_4,311). happensAt(inactive(fcgt_4),311).
orientation(fcgt_3,164,312). coords(fcgt_3,98,188,312).  visible(fcgt_3,312). happensAt(inactive(fcgt_3),312). orientation(fcgt_4,166,312). coords(fcgt_4,82,206,312).  visible(fcgt_4,312). happensAt(inactive(fcgt_4),312).
orientation(fcgt_3,164,313). coords(fcgt_3,98,188,313).  visible(fcgt_3,313). happensAt(inactive(fcgt_3),313). orientation(fcgt_4,166,313). coords(fcgt_4,82,206,313).  visible(fcgt_4,313). happensAt(inactive(fcgt_4),313).
orientation(fcgt_3,164,314). coords(fcgt_3,98,188,314).  visible(fcgt_3,314). happensAt(inactive(fcgt_3),314). orientation(fcgt_4,166,314). coords(fcgt_4,82,206,314).  visible(fcgt_4,314). happensAt(inactive(fcgt_4),314).
orientation(fcgt_3,164,315). coords(fcgt_3,98,188,315).  visible(fcgt_3,315). happensAt(inactive(fcgt_3),315). orientation(fcgt_4,166,315). coords(fcgt_4,82,206,315).  visible(fcgt_4,315). happensAt(inactive(fcgt_4),315).
orientation(fcgt_3,164,316). coords(fcgt_3,98,188,316).  visible(fcgt_3,316). happensAt(inactive(fcgt_3),316). orientation(fcgt_4,166,316). coords(fcgt_4,82,206,316).  visible(fcgt_4,316). happensAt(inactive(fcgt_4),316).
orientation(fcgt_3,164,317). coords(fcgt_3,98,188,317).  visible(fcgt_3,317). happensAt(inactive(fcgt_3),317). orientation(fcgt_4,166,317). coords(fcgt_4,82,206,317).  visible(fcgt_4,317). happensAt(inactive(fcgt_4),317).
orientation(fcgt_3,164,318). coords(fcgt_3,99,188,318).  visible(fcgt_3,318). happensAt(inactive(fcgt_3),318). orientation(fcgt_4,166,318). coords(fcgt_4,82,206,318).  visible(fcgt_4,318). happensAt(inactive(fcgt_4),318).
orientation(fcgt_3,164,319). coords(fcgt_3,98,187,319).  visible(fcgt_3,319). happensAt(inactive(fcgt_3),319). orientation(fcgt_4,166,319). coords(fcgt_4,82,206,319).  visible(fcgt_4,319). happensAt(inactive(fcgt_4),319).
orientation(fcgt_3,164,320). coords(fcgt_3,99,187,320).  visible(fcgt_3,320). happensAt(inactive(fcgt_3),320). orientation(fcgt_4,166,320). coords(fcgt_4,82,206,320).  visible(fcgt_4,320). happensAt(inactive(fcgt_4),320).
orientation(fcgt_3,164,321). coords(fcgt_3,98,187,321).  visible(fcgt_3,321). happensAt(inactive(fcgt_3),321). orientation(fcgt_4,166,321). coords(fcgt_4,82,206,321).  visible(fcgt_4,321). happensAt(inactive(fcgt_4),321).
orientation(fcgt_3,164,322). coords(fcgt_3,98,187,322).  visible(fcgt_3,322). happensAt(inactive(fcgt_3),322). orientation(fcgt_4,166,322). coords(fcgt_4,82,206,322).  visible(fcgt_4,322). happensAt(inactive(fcgt_4),322).
orientation(fcgt_3,164,323). coords(fcgt_3,98,187,323).  visible(fcgt_3,323). happensAt(inactive(fcgt_3),323). orientation(fcgt_4,166,323). coords(fcgt_4,82,206,323).  visible(fcgt_4,323). happensAt(inactive(fcgt_4),323).
orientation(fcgt_3,164,324). coords(fcgt_3,98,187,324).  visible(fcgt_3,324). happensAt(inactive(fcgt_3),324). orientation(fcgt_4,166,324). coords(fcgt_4,82,206,324).  visible(fcgt_4,324). happensAt(inactive(fcgt_4),324).
#show holdsAt/2.