
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

	orientation(fomdgt3_1,150,318). coords(fomdgt3_1,54,100,318).  visible(fomdgt3_1,318). happensAt(inactive(fomdgt3_1),318). orientation(fomdgt3_6,130,318). coords(fomdgt3_6,212,110,318).  visible(fomdgt3_6,318). happensAt(walking(fomdgt3_6),318).
orientation(fomdgt3_1,150,319). coords(fomdgt3_1,54,100,319).  visible(fomdgt3_1,319). happensAt(inactive(fomdgt3_1),319). orientation(fomdgt3_6,130,319). coords(fomdgt3_6,212,110,319).  visible(fomdgt3_6,319). happensAt(walking(fomdgt3_6),319).
orientation(fomdgt3_1,150,320). coords(fomdgt3_1,54,100,320).  visible(fomdgt3_1,320). happensAt(inactive(fomdgt3_1),320). orientation(fomdgt3_6,130,320). coords(fomdgt3_6,212,110,320).  visible(fomdgt3_6,320). happensAt(walking(fomdgt3_6),320).
orientation(fomdgt3_1,150,321). coords(fomdgt3_1,54,100,321).  visible(fomdgt3_1,321). happensAt(inactive(fomdgt3_1),321). orientation(fomdgt3_6,130,321). coords(fomdgt3_6,212,110,321).  visible(fomdgt3_6,321). happensAt(walking(fomdgt3_6),321).
orientation(fomdgt3_1,150,322). coords(fomdgt3_1,54,100,322).  visible(fomdgt3_1,322). happensAt(inactive(fomdgt3_1),322). orientation(fomdgt3_6,130,322). coords(fomdgt3_6,212,112,322).  visible(fomdgt3_6,322). happensAt(walking(fomdgt3_6),322).
orientation(fomdgt3_1,150,323). coords(fomdgt3_1,54,100,323).  visible(fomdgt3_1,323). happensAt(inactive(fomdgt3_1),323). orientation(fomdgt3_6,130,323). coords(fomdgt3_6,212,112,323).  visible(fomdgt3_6,323). happensAt(walking(fomdgt3_6),323).
orientation(fomdgt3_1,150,324). coords(fomdgt3_1,54,100,324).  visible(fomdgt3_1,324). happensAt(inactive(fomdgt3_1),324). orientation(fomdgt3_6,130,324). coords(fomdgt3_6,212,112,324).  visible(fomdgt3_6,324). happensAt(walking(fomdgt3_6),324).
orientation(fomdgt3_1,150,325). coords(fomdgt3_1,54,100,325).  visible(fomdgt3_1,325). happensAt(inactive(fomdgt3_1),325). orientation(fomdgt3_6,130,325). coords(fomdgt3_6,212,112,325).  visible(fomdgt3_6,325). happensAt(walking(fomdgt3_6),325).
orientation(fomdgt3_1,150,326). coords(fomdgt3_1,54,100,326).  visible(fomdgt3_1,326). happensAt(inactive(fomdgt3_1),326). orientation(fomdgt3_6,130,326). coords(fomdgt3_6,212,112,326).  visible(fomdgt3_6,326). happensAt(walking(fomdgt3_6),326).
orientation(fomdgt3_1,150,327). coords(fomdgt3_1,54,100,327).  visible(fomdgt3_1,327). happensAt(inactive(fomdgt3_1),327). orientation(fomdgt3_6,130,327). coords(fomdgt3_6,212,112,327).  visible(fomdgt3_6,327). happensAt(walking(fomdgt3_6),327).
orientation(fomdgt3_1,150,328). coords(fomdgt3_1,54,100,328).  visible(fomdgt3_1,328). happensAt(inactive(fomdgt3_1),328). orientation(fomdgt3_6,130,328). coords(fomdgt3_6,212,111,328).  visible(fomdgt3_6,328). happensAt(walking(fomdgt3_6),328).
orientation(fomdgt3_1,150,329). coords(fomdgt3_1,54,100,329).  visible(fomdgt3_1,329). happensAt(inactive(fomdgt3_1),329). orientation(fomdgt3_6,130,329). coords(fomdgt3_6,212,111,329).  visible(fomdgt3_6,329). happensAt(walking(fomdgt3_6),329).
orientation(fomdgt3_1,150,330). coords(fomdgt3_1,54,100,330).  visible(fomdgt3_1,330). happensAt(inactive(fomdgt3_1),330). orientation(fomdgt3_6,130,330). coords(fomdgt3_6,212,111,330).  visible(fomdgt3_6,330). happensAt(walking(fomdgt3_6),330).
orientation(fomdgt3_1,150,331). coords(fomdgt3_1,54,100,331).  visible(fomdgt3_1,331). happensAt(inactive(fomdgt3_1),331). orientation(fomdgt3_6,130,331). coords(fomdgt3_6,212,111,331).  visible(fomdgt3_6,331). happensAt(walking(fomdgt3_6),331).
orientation(fomdgt3_1,150,332). coords(fomdgt3_1,54,100,332).  visible(fomdgt3_1,332). happensAt(inactive(fomdgt3_1),332). orientation(fomdgt3_6,130,332). coords(fomdgt3_6,212,111,332).  visible(fomdgt3_6,332). happensAt(walking(fomdgt3_6),332).
orientation(fomdgt3_1,150,333). coords(fomdgt3_1,54,100,333).  visible(fomdgt3_1,333). happensAt(inactive(fomdgt3_1),333). orientation(fomdgt3_6,130,333). coords(fomdgt3_6,212,111,333).  visible(fomdgt3_6,333). happensAt(walking(fomdgt3_6),333).
orientation(fomdgt3_1,150,334). coords(fomdgt3_1,54,100,334).  visible(fomdgt3_1,334). happensAt(inactive(fomdgt3_1),334). orientation(fomdgt3_6,130,334). coords(fomdgt3_6,212,111,334).  visible(fomdgt3_6,334). happensAt(walking(fomdgt3_6),334).
orientation(fomdgt3_1,150,335). coords(fomdgt3_1,54,100,335).  visible(fomdgt3_1,335). happensAt(inactive(fomdgt3_1),335). orientation(fomdgt3_6,130,335). coords(fomdgt3_6,212,111,335).  visible(fomdgt3_6,335). happensAt(walking(fomdgt3_6),335).
orientation(fomdgt3_1,150,336). coords(fomdgt3_1,54,100,336).  visible(fomdgt3_1,336). happensAt(inactive(fomdgt3_1),336). orientation(fomdgt3_6,130,336). coords(fomdgt3_6,212,111,336).  visible(fomdgt3_6,336). happensAt(walking(fomdgt3_6),336).
orientation(fomdgt3_1,150,337). coords(fomdgt3_1,54,100,337).  visible(fomdgt3_1,337). happensAt(inactive(fomdgt3_1),337). orientation(fomdgt3_6,130,337). coords(fomdgt3_6,212,111,337).  visible(fomdgt3_6,337). happensAt(walking(fomdgt3_6),337).
orientation(fomdgt3_1,150,338). coords(fomdgt3_1,54,100,338).  visible(fomdgt3_1,338). happensAt(inactive(fomdgt3_1),338). orientation(fomdgt3_6,130,338). coords(fomdgt3_6,212,110,338).  visible(fomdgt3_6,338). happensAt(walking(fomdgt3_6),338).
orientation(fomdgt3_1,150,339). coords(fomdgt3_1,54,100,339).  visible(fomdgt3_1,339). happensAt(inactive(fomdgt3_1),339). orientation(fomdgt3_6,130,339). coords(fomdgt3_6,212,110,339).  visible(fomdgt3_6,339). happensAt(walking(fomdgt3_6),339).
orientation(fomdgt3_1,150,340). coords(fomdgt3_1,54,100,340).  visible(fomdgt3_1,340). happensAt(inactive(fomdgt3_1),340). orientation(fomdgt3_6,130,340). coords(fomdgt3_6,214,110,340).  visible(fomdgt3_6,340). happensAt(walking(fomdgt3_6),340).
orientation(fomdgt3_1,150,341). coords(fomdgt3_1,54,100,341).  visible(fomdgt3_1,341). happensAt(inactive(fomdgt3_1),341). orientation(fomdgt3_6,130,341). coords(fomdgt3_6,214,110,341).  visible(fomdgt3_6,341). happensAt(walking(fomdgt3_6),341).
orientation(fomdgt3_1,150,342). coords(fomdgt3_1,54,100,342).  visible(fomdgt3_1,342). happensAt(inactive(fomdgt3_1),342). orientation(fomdgt3_6,130,342). coords(fomdgt3_6,217,110,342).  visible(fomdgt3_6,342). happensAt(walking(fomdgt3_6),342).
#show holdsAt/2.