
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

	orientation(fcgt_2,150,318). coords(fcgt_2,95,129,318).  visible(fcgt_2,318). happensAt(active(fcgt_2),318). orientation(fcgt_3,164,318). coords(fcgt_3,99,188,318).  visible(fcgt_3,318). happensAt(inactive(fcgt_3),318).
orientation(fcgt_2,150,319). coords(fcgt_2,94,129,319).  visible(fcgt_2,319). happensAt(active(fcgt_2),319). orientation(fcgt_3,164,319). coords(fcgt_3,98,187,319).  visible(fcgt_3,319). happensAt(inactive(fcgt_3),319).
orientation(fcgt_2,150,320). coords(fcgt_2,94,129,320).  visible(fcgt_2,320). happensAt(active(fcgt_2),320). orientation(fcgt_3,164,320). coords(fcgt_3,99,187,320).  visible(fcgt_3,320). happensAt(inactive(fcgt_3),320).
orientation(fcgt_2,150,321). coords(fcgt_2,94,129,321).  visible(fcgt_2,321). happensAt(active(fcgt_2),321). orientation(fcgt_3,164,321). coords(fcgt_3,98,187,321).  visible(fcgt_3,321). happensAt(inactive(fcgt_3),321).
orientation(fcgt_2,150,322). coords(fcgt_2,94,129,322).  visible(fcgt_2,322). happensAt(active(fcgt_2),322). orientation(fcgt_3,164,322). coords(fcgt_3,98,187,322).  visible(fcgt_3,322). happensAt(inactive(fcgt_3),322).
orientation(fcgt_2,150,323). coords(fcgt_2,94,129,323).  visible(fcgt_2,323). happensAt(active(fcgt_2),323). orientation(fcgt_3,164,323). coords(fcgt_3,98,187,323).  visible(fcgt_3,323). happensAt(inactive(fcgt_3),323).
orientation(fcgt_2,150,324). coords(fcgt_2,94,129,324).  visible(fcgt_2,324). happensAt(active(fcgt_2),324). orientation(fcgt_3,164,324). coords(fcgt_3,98,187,324).  visible(fcgt_3,324). happensAt(inactive(fcgt_3),324).
orientation(fcgt_2,150,325). coords(fcgt_2,93,128,325).  visible(fcgt_2,325). happensAt(active(fcgt_2),325). orientation(fcgt_3,164,325). coords(fcgt_3,98,187,325).  visible(fcgt_3,325). happensAt(inactive(fcgt_3),325).
orientation(fcgt_2,146,326). coords(fcgt_2,93,128,326).  visible(fcgt_2,326). happensAt(active(fcgt_2),326). orientation(fcgt_3,164,326). coords(fcgt_3,98,187,326).  visible(fcgt_3,326). happensAt(inactive(fcgt_3),326).
orientation(fcgt_2,146,327). coords(fcgt_2,92,127,327).  visible(fcgt_2,327). happensAt(active(fcgt_2),327). orientation(fcgt_3,164,327). coords(fcgt_3,98,187,327).  visible(fcgt_3,327). happensAt(inactive(fcgt_3),327).
orientation(fcgt_2,146,328). coords(fcgt_2,91,127,328).  visible(fcgt_2,328). happensAt(active(fcgt_2),328). orientation(fcgt_3,164,328). coords(fcgt_3,98,187,328).  visible(fcgt_3,328). happensAt(inactive(fcgt_3),328).
orientation(fcgt_2,146,329). coords(fcgt_2,91,126,329).  visible(fcgt_2,329). happensAt(active(fcgt_2),329). orientation(fcgt_3,164,329). coords(fcgt_3,98,187,329).  visible(fcgt_3,329). happensAt(inactive(fcgt_3),329).
orientation(fcgt_2,146,330). coords(fcgt_2,91,125,330).  visible(fcgt_2,330). happensAt(active(fcgt_2),330). orientation(fcgt_3,164,330). coords(fcgt_3,98,187,330).  visible(fcgt_3,330). happensAt(inactive(fcgt_3),330).
orientation(fcgt_2,146,331). coords(fcgt_2,91,124,331).  visible(fcgt_2,331). happensAt(active(fcgt_2),331). orientation(fcgt_3,164,331). coords(fcgt_3,98,187,331).  visible(fcgt_3,331). happensAt(inactive(fcgt_3),331).
orientation(fcgt_2,146,332). coords(fcgt_2,91,124,332).  visible(fcgt_2,332). happensAt(active(fcgt_2),332). orientation(fcgt_3,164,332). coords(fcgt_3,98,187,332).  visible(fcgt_3,332). happensAt(inactive(fcgt_3),332).
orientation(fcgt_2,146,333). coords(fcgt_2,92,124,333).  visible(fcgt_2,333). happensAt(active(fcgt_2),333). orientation(fcgt_3,164,333). coords(fcgt_3,98,187,333).  visible(fcgt_3,333). happensAt(inactive(fcgt_3),333).
orientation(fcgt_2,146,334). coords(fcgt_2,93,123,334).  visible(fcgt_2,334). happensAt(active(fcgt_2),334). orientation(fcgt_3,164,334). coords(fcgt_3,98,187,334).  visible(fcgt_3,334). happensAt(inactive(fcgt_3),334).
orientation(fcgt_2,146,335). coords(fcgt_2,94,123,335).  visible(fcgt_2,335). happensAt(active(fcgt_2),335). orientation(fcgt_3,164,335). coords(fcgt_3,98,187,335).  visible(fcgt_3,335). happensAt(inactive(fcgt_3),335).
orientation(fcgt_2,155,336). coords(fcgt_2,94,121,336).  visible(fcgt_2,336). happensAt(active(fcgt_2),336). orientation(fcgt_3,164,336). coords(fcgt_3,98,187,336).  visible(fcgt_3,336). happensAt(inactive(fcgt_3),336).
orientation(fcgt_2,155,337). coords(fcgt_2,94,120,337).  visible(fcgt_2,337). happensAt(active(fcgt_2),337). orientation(fcgt_3,164,337). coords(fcgt_3,98,187,337).  visible(fcgt_3,337). happensAt(inactive(fcgt_3),337).
orientation(fcgt_2,155,338). coords(fcgt_2,94,120,338).  visible(fcgt_2,338). happensAt(active(fcgt_2),338). orientation(fcgt_3,164,338). coords(fcgt_3,98,187,338).  visible(fcgt_3,338). happensAt(inactive(fcgt_3),338).
orientation(fcgt_2,155,339). coords(fcgt_2,94,119,339).  visible(fcgt_2,339). happensAt(active(fcgt_2),339). orientation(fcgt_3,164,339). coords(fcgt_3,98,187,339).  visible(fcgt_3,339). happensAt(inactive(fcgt_3),339).
orientation(fcgt_2,155,340). coords(fcgt_2,94,117,340).  visible(fcgt_2,340). happensAt(active(fcgt_2),340). orientation(fcgt_3,164,340). coords(fcgt_3,98,187,340).  visible(fcgt_3,340). happensAt(inactive(fcgt_3),340).
orientation(fcgt_2,155,341). coords(fcgt_2,94,117,341).  visible(fcgt_2,341). happensAt(active(fcgt_2),341). orientation(fcgt_3,164,341). coords(fcgt_3,98,187,341).  visible(fcgt_3,341). happensAt(inactive(fcgt_3),341).
orientation(fcgt_2,155,342). coords(fcgt_2,94,116,342).  visible(fcgt_2,342). happensAt(active(fcgt_2),342). orientation(fcgt_3,164,342). coords(fcgt_3,98,187,342).  visible(fcgt_3,342). happensAt(inactive(fcgt_3),342).
#show holdsAt/2.