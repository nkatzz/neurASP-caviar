
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

	orientation(fomdgt3_2,80,318). coords(fomdgt3_2,244,110,318).  visible(fomdgt3_2,318). happensAt(walking(fomdgt3_2),318). orientation(fomdgt3_6,130,318). coords(fomdgt3_6,212,110,318).  visible(fomdgt3_6,318). happensAt(walking(fomdgt3_6),318).
orientation(fomdgt3_2,80,319). coords(fomdgt3_2,249,107,319).  visible(fomdgt3_2,319). happensAt(walking(fomdgt3_2),319). orientation(fomdgt3_6,130,319). coords(fomdgt3_6,212,110,319).  visible(fomdgt3_6,319). happensAt(walking(fomdgt3_6),319).
orientation(fomdgt3_2,80,320). coords(fomdgt3_2,249,104,320).  visible(fomdgt3_2,320). happensAt(walking(fomdgt3_2),320). orientation(fomdgt3_6,130,320). coords(fomdgt3_6,212,110,320).  visible(fomdgt3_6,320). happensAt(walking(fomdgt3_6),320).
orientation(fomdgt3_2,80,321). coords(fomdgt3_2,250,104,321).  visible(fomdgt3_2,321). happensAt(walking(fomdgt3_2),321). orientation(fomdgt3_6,130,321). coords(fomdgt3_6,212,110,321).  visible(fomdgt3_6,321). happensAt(walking(fomdgt3_6),321).
orientation(fomdgt3_2,70,322). coords(fomdgt3_2,248,102,322).  visible(fomdgt3_2,322). happensAt(walking(fomdgt3_2),322). orientation(fomdgt3_6,130,322). coords(fomdgt3_6,212,112,322).  visible(fomdgt3_6,322). happensAt(walking(fomdgt3_6),322).
orientation(fomdgt3_2,70,323). coords(fomdgt3_2,248,102,323).  visible(fomdgt3_2,323). happensAt(walking(fomdgt3_2),323). orientation(fomdgt3_6,130,323). coords(fomdgt3_6,212,112,323).  visible(fomdgt3_6,323). happensAt(walking(fomdgt3_6),323).
orientation(fomdgt3_2,60,324). coords(fomdgt3_2,248,98,324).  visible(fomdgt3_2,324). happensAt(walking(fomdgt3_2),324). orientation(fomdgt3_6,130,324). coords(fomdgt3_6,212,112,324).  visible(fomdgt3_6,324). happensAt(walking(fomdgt3_6),324).
orientation(fomdgt3_2,60,325). coords(fomdgt3_2,248,97,325).  visible(fomdgt3_2,325). happensAt(walking(fomdgt3_2),325). orientation(fomdgt3_6,130,325). coords(fomdgt3_6,212,112,325).  visible(fomdgt3_6,325). happensAt(walking(fomdgt3_6),325).
orientation(fomdgt3_2,60,326). coords(fomdgt3_2,251,94,326).  visible(fomdgt3_2,326). happensAt(walking(fomdgt3_2),326). orientation(fomdgt3_6,130,326). coords(fomdgt3_6,212,112,326).  visible(fomdgt3_6,326). happensAt(walking(fomdgt3_6),326).
orientation(fomdgt3_2,60,327). coords(fomdgt3_2,253,91,327).  visible(fomdgt3_2,327). happensAt(walking(fomdgt3_2),327). orientation(fomdgt3_6,130,327). coords(fomdgt3_6,212,112,327).  visible(fomdgt3_6,327). happensAt(walking(fomdgt3_6),327).
orientation(fomdgt3_2,60,328). coords(fomdgt3_2,252,89,328).  visible(fomdgt3_2,328). happensAt(walking(fomdgt3_2),328). orientation(fomdgt3_6,130,328). coords(fomdgt3_6,212,111,328).  visible(fomdgt3_6,328). happensAt(walking(fomdgt3_6),328).
orientation(fomdgt3_2,30,329). coords(fomdgt3_2,252,89,329).  visible(fomdgt3_2,329). happensAt(walking(fomdgt3_2),329). orientation(fomdgt3_6,130,329). coords(fomdgt3_6,212,111,329).  visible(fomdgt3_6,329). happensAt(walking(fomdgt3_6),329).
orientation(fomdgt3_2,30,330). coords(fomdgt3_2,254,88,330).  visible(fomdgt3_2,330). happensAt(walking(fomdgt3_2),330). orientation(fomdgt3_6,130,330). coords(fomdgt3_6,212,111,330).  visible(fomdgt3_6,330). happensAt(walking(fomdgt3_6),330).
orientation(fomdgt3_2,30,331). coords(fomdgt3_2,254,88,331).  visible(fomdgt3_2,331). happensAt(walking(fomdgt3_2),331). orientation(fomdgt3_6,130,331). coords(fomdgt3_6,212,111,331).  visible(fomdgt3_6,331). happensAt(walking(fomdgt3_6),331).
orientation(fomdgt3_2,30,332). coords(fomdgt3_2,257,87,332).  visible(fomdgt3_2,332). happensAt(walking(fomdgt3_2),332). orientation(fomdgt3_6,130,332). coords(fomdgt3_6,212,111,332).  visible(fomdgt3_6,332). happensAt(walking(fomdgt3_6),332).
orientation(fomdgt3_2,30,333). coords(fomdgt3_2,257,86,333).  visible(fomdgt3_2,333). happensAt(walking(fomdgt3_2),333). orientation(fomdgt3_6,130,333). coords(fomdgt3_6,212,111,333).  visible(fomdgt3_6,333). happensAt(walking(fomdgt3_6),333).
orientation(fomdgt3_2,30,334). coords(fomdgt3_2,257,86,334).  visible(fomdgt3_2,334). happensAt(walking(fomdgt3_2),334). orientation(fomdgt3_6,130,334). coords(fomdgt3_6,212,111,334).  visible(fomdgt3_6,334). happensAt(walking(fomdgt3_6),334).
orientation(fomdgt3_2,30,335). coords(fomdgt3_2,261,83,335).  visible(fomdgt3_2,335). happensAt(walking(fomdgt3_2),335). orientation(fomdgt3_6,130,335). coords(fomdgt3_6,212,111,335).  visible(fomdgt3_6,335). happensAt(walking(fomdgt3_6),335).
orientation(fomdgt3_2,30,336). coords(fomdgt3_2,262,81,336).  visible(fomdgt3_2,336). happensAt(walking(fomdgt3_2),336). orientation(fomdgt3_6,130,336). coords(fomdgt3_6,212,111,336).  visible(fomdgt3_6,336). happensAt(walking(fomdgt3_6),336).
orientation(fomdgt3_2,30,337). coords(fomdgt3_2,263,81,337).  visible(fomdgt3_2,337). happensAt(walking(fomdgt3_2),337). orientation(fomdgt3_6,130,337). coords(fomdgt3_6,212,111,337).  visible(fomdgt3_6,337). happensAt(walking(fomdgt3_6),337).
orientation(fomdgt3_2,30,338). coords(fomdgt3_2,263,81,338).  visible(fomdgt3_2,338). happensAt(walking(fomdgt3_2),338). orientation(fomdgt3_6,130,338). coords(fomdgt3_6,212,110,338).  visible(fomdgt3_6,338). happensAt(walking(fomdgt3_6),338).
orientation(fomdgt3_2,30,339). coords(fomdgt3_2,263,81,339).  visible(fomdgt3_2,339). happensAt(walking(fomdgt3_2),339). orientation(fomdgt3_6,130,339). coords(fomdgt3_6,212,110,339).  visible(fomdgt3_6,339). happensAt(walking(fomdgt3_6),339).
orientation(fomdgt3_2,30,340). coords(fomdgt3_2,263,81,340).  visible(fomdgt3_2,340). happensAt(walking(fomdgt3_2),340). orientation(fomdgt3_6,130,340). coords(fomdgt3_6,214,110,340).  visible(fomdgt3_6,340). happensAt(walking(fomdgt3_6),340).
orientation(fomdgt3_2,30,341). coords(fomdgt3_2,263,81,341).  visible(fomdgt3_2,341). happensAt(walking(fomdgt3_2),341). orientation(fomdgt3_6,130,341). coords(fomdgt3_6,214,110,341).  visible(fomdgt3_6,341). happensAt(walking(fomdgt3_6),341).
orientation(fomdgt3_2,30,342). coords(fomdgt3_2,274,81,342).  visible(fomdgt3_2,342). happensAt(walking(fomdgt3_2),342). orientation(fomdgt3_6,130,342). coords(fomdgt3_6,217,110,342).  visible(fomdgt3_6,342). happensAt(walking(fomdgt3_6),342).
#show holdsAt/2.