
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
orientation(fcgt_3,164,325). coords(fcgt_3,98,187,325).  visible(fcgt_3,325). happensAt(inactive(fcgt_3),325). orientation(fcgt_4,166,325). coords(fcgt_4,82,206,325).  visible(fcgt_4,325). happensAt(inactive(fcgt_4),325).
orientation(fcgt_3,164,326). coords(fcgt_3,98,187,326).  visible(fcgt_3,326). happensAt(inactive(fcgt_3),326). orientation(fcgt_4,166,326). coords(fcgt_4,82,206,326).  visible(fcgt_4,326). happensAt(inactive(fcgt_4),326).
orientation(fcgt_3,164,327). coords(fcgt_3,98,187,327).  visible(fcgt_3,327). happensAt(inactive(fcgt_3),327). orientation(fcgt_4,166,327). coords(fcgt_4,82,206,327).  visible(fcgt_4,327). happensAt(inactive(fcgt_4),327).
orientation(fcgt_3,164,328). coords(fcgt_3,98,187,328).  visible(fcgt_3,328). happensAt(inactive(fcgt_3),328). orientation(fcgt_4,166,328). coords(fcgt_4,82,206,328).  visible(fcgt_4,328). happensAt(inactive(fcgt_4),328).
orientation(fcgt_3,164,329). coords(fcgt_3,98,187,329).  visible(fcgt_3,329). happensAt(inactive(fcgt_3),329). orientation(fcgt_4,166,329). coords(fcgt_4,82,206,329).  visible(fcgt_4,329). happensAt(inactive(fcgt_4),329).
orientation(fcgt_3,164,330). coords(fcgt_3,98,187,330).  visible(fcgt_3,330). happensAt(inactive(fcgt_3),330). orientation(fcgt_4,166,330). coords(fcgt_4,82,206,330).  visible(fcgt_4,330). happensAt(inactive(fcgt_4),330).
orientation(fcgt_3,164,331). coords(fcgt_3,98,187,331).  visible(fcgt_3,331). happensAt(inactive(fcgt_3),331). orientation(fcgt_4,166,331). coords(fcgt_4,82,206,331).  visible(fcgt_4,331). happensAt(inactive(fcgt_4),331).
orientation(fcgt_3,164,332). coords(fcgt_3,98,187,332).  visible(fcgt_3,332). happensAt(inactive(fcgt_3),332). orientation(fcgt_4,166,332). coords(fcgt_4,82,206,332).  visible(fcgt_4,332). happensAt(inactive(fcgt_4),332).
orientation(fcgt_3,164,333). coords(fcgt_3,98,187,333).  visible(fcgt_3,333). happensAt(inactive(fcgt_3),333). orientation(fcgt_4,166,333). coords(fcgt_4,82,206,333).  visible(fcgt_4,333). happensAt(inactive(fcgt_4),333).
orientation(fcgt_3,164,334). coords(fcgt_3,98,187,334).  visible(fcgt_3,334). happensAt(inactive(fcgt_3),334). orientation(fcgt_4,166,334). coords(fcgt_4,82,206,334).  visible(fcgt_4,334). happensAt(inactive(fcgt_4),334).
orientation(fcgt_3,164,335). coords(fcgt_3,98,187,335).  visible(fcgt_3,335). happensAt(inactive(fcgt_3),335). orientation(fcgt_4,166,335). coords(fcgt_4,82,206,335).  visible(fcgt_4,335). happensAt(inactive(fcgt_4),335).
orientation(fcgt_3,164,336). coords(fcgt_3,98,187,336).  visible(fcgt_3,336). happensAt(inactive(fcgt_3),336). orientation(fcgt_4,166,336). coords(fcgt_4,82,206,336).  visible(fcgt_4,336). happensAt(inactive(fcgt_4),336).
#show holdsAt/2.