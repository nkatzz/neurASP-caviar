
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

	orientation(ms3ggt_1,162,320). coords(ms3ggt_1,90,166,320).  visible(ms3ggt_1,320). happensAt(walking(ms3ggt_1),320). orientation(ms3ggt_2,13,320). coords(ms3ggt_2,233,280,320).  happensAt(appear(ms3ggt_2),320). happensAt(walking(ms3ggt_2),320).
orientation(ms3ggt_1,162,321). coords(ms3ggt_1,91,166,321).  visible(ms3ggt_1,321). happensAt(walking(ms3ggt_1),321). orientation(ms3ggt_2,13,321). coords(ms3ggt_2,233,278,321).  visible(ms3ggt_2,321). happensAt(walking(ms3ggt_2),321).
orientation(ms3ggt_1,162,322). coords(ms3ggt_1,92,166,322).  visible(ms3ggt_1,322). happensAt(walking(ms3ggt_1),322). orientation(ms3ggt_2,13,322). coords(ms3ggt_2,220,276,322).  visible(ms3ggt_2,322). happensAt(walking(ms3ggt_2),322).
orientation(ms3ggt_1,162,323). coords(ms3ggt_1,93,166,323).  visible(ms3ggt_1,323). happensAt(walking(ms3ggt_1),323). orientation(ms3ggt_2,13,323). coords(ms3ggt_2,219,275,323).  visible(ms3ggt_2,323). happensAt(walking(ms3ggt_2),323).
orientation(ms3ggt_1,162,324). coords(ms3ggt_1,93,166,324).  visible(ms3ggt_1,324). happensAt(walking(ms3ggt_1),324). orientation(ms3ggt_2,13,324). coords(ms3ggt_2,218,275,324).  visible(ms3ggt_2,324). happensAt(walking(ms3ggt_2),324).
orientation(ms3ggt_1,162,325). coords(ms3ggt_1,93,167,325).  visible(ms3ggt_1,325). happensAt(walking(ms3ggt_1),325). orientation(ms3ggt_2,13,325). coords(ms3ggt_2,216,275,325).  visible(ms3ggt_2,325). happensAt(walking(ms3ggt_2),325).
orientation(ms3ggt_1,162,326). coords(ms3ggt_1,93,167,326).  visible(ms3ggt_1,326). happensAt(walking(ms3ggt_1),326). orientation(ms3ggt_2,13,326). coords(ms3ggt_2,216,275,326).  visible(ms3ggt_2,326). happensAt(walking(ms3ggt_2),326).
orientation(ms3ggt_1,162,327). coords(ms3ggt_1,93,167,327).  visible(ms3ggt_1,327). happensAt(walking(ms3ggt_1),327). orientation(ms3ggt_2,13,327). coords(ms3ggt_2,216,275,327).  visible(ms3ggt_2,327). happensAt(walking(ms3ggt_2),327).
orientation(ms3ggt_1,162,328). coords(ms3ggt_1,93,167,328).  visible(ms3ggt_1,328). happensAt(walking(ms3ggt_1),328). orientation(ms3ggt_2,13,328). coords(ms3ggt_2,216,275,328).  visible(ms3ggt_2,328). happensAt(walking(ms3ggt_2),328).
orientation(ms3ggt_1,156,329). coords(ms3ggt_1,94,167,329).  visible(ms3ggt_1,329). happensAt(walking(ms3ggt_1),329). orientation(ms3ggt_2,13,329). coords(ms3ggt_2,216,275,329).  visible(ms3ggt_2,329). happensAt(walking(ms3ggt_2),329).
orientation(ms3ggt_1,156,330). coords(ms3ggt_1,95,167,330).  visible(ms3ggt_1,330). happensAt(walking(ms3ggt_1),330). orientation(ms3ggt_2,13,330). coords(ms3ggt_2,216,275,330).  visible(ms3ggt_2,330). happensAt(walking(ms3ggt_2),330).
orientation(ms3ggt_1,156,331). coords(ms3ggt_1,95,167,331).  visible(ms3ggt_1,331). happensAt(walking(ms3ggt_1),331). orientation(ms3ggt_2,13,331). coords(ms3ggt_2,216,273,331).  visible(ms3ggt_2,331). happensAt(walking(ms3ggt_2),331).
orientation(ms3ggt_1,156,332). coords(ms3ggt_1,97,167,332).  visible(ms3ggt_1,332). happensAt(walking(ms3ggt_1),332). orientation(ms3ggt_2,13,332). coords(ms3ggt_2,214,273,332).  visible(ms3ggt_2,332). happensAt(walking(ms3ggt_2),332).
orientation(ms3ggt_1,156,333). coords(ms3ggt_1,98,167,333).  visible(ms3ggt_1,333). happensAt(walking(ms3ggt_1),333). orientation(ms3ggt_2,13,333). coords(ms3ggt_2,213,271,333).  visible(ms3ggt_2,333). happensAt(walking(ms3ggt_2),333).
orientation(ms3ggt_1,156,334). coords(ms3ggt_1,100,167,334).  visible(ms3ggt_1,334). happensAt(walking(ms3ggt_1),334). orientation(ms3ggt_2,13,334). coords(ms3ggt_2,213,269,334).  visible(ms3ggt_2,334). happensAt(walking(ms3ggt_2),334).
orientation(ms3ggt_1,156,335). coords(ms3ggt_1,101,166,335).  visible(ms3ggt_1,335). happensAt(walking(ms3ggt_1),335). orientation(ms3ggt_2,13,335). coords(ms3ggt_2,213,265,335).  visible(ms3ggt_2,335). happensAt(walking(ms3ggt_2),335).
orientation(ms3ggt_1,156,336). coords(ms3ggt_1,103,166,336).  visible(ms3ggt_1,336). happensAt(walking(ms3ggt_1),336). orientation(ms3ggt_2,13,336). coords(ms3ggt_2,213,264,336).  visible(ms3ggt_2,336). happensAt(walking(ms3ggt_2),336).
orientation(ms3ggt_1,156,337). coords(ms3ggt_1,103,166,337).  visible(ms3ggt_1,337). happensAt(walking(ms3ggt_1),337). orientation(ms3ggt_2,13,337). coords(ms3ggt_2,213,262,337).  visible(ms3ggt_2,337). happensAt(walking(ms3ggt_2),337).
orientation(ms3ggt_1,156,338). coords(ms3ggt_1,104,166,338).  visible(ms3ggt_1,338). happensAt(walking(ms3ggt_1),338). orientation(ms3ggt_2,13,338). coords(ms3ggt_2,212,260,338).  visible(ms3ggt_2,338). happensAt(walking(ms3ggt_2),338).
orientation(ms3ggt_1,156,339). coords(ms3ggt_1,104,166,339).  visible(ms3ggt_1,339). happensAt(walking(ms3ggt_1),339). orientation(ms3ggt_2,13,339). coords(ms3ggt_2,212,260,339).  visible(ms3ggt_2,339). happensAt(walking(ms3ggt_2),339).
orientation(ms3ggt_1,156,340). coords(ms3ggt_1,105,166,340).  visible(ms3ggt_1,340). happensAt(walking(ms3ggt_1),340). orientation(ms3ggt_2,13,340). coords(ms3ggt_2,209,258,340).  visible(ms3ggt_2,340). happensAt(walking(ms3ggt_2),340).
orientation(ms3ggt_1,156,341). coords(ms3ggt_1,105,166,341).  visible(ms3ggt_1,341). happensAt(walking(ms3ggt_1),341). orientation(ms3ggt_2,13,341). coords(ms3ggt_2,208,256,341).  visible(ms3ggt_2,341). happensAt(walking(ms3ggt_2),341).
orientation(ms3ggt_1,156,342). coords(ms3ggt_1,105,166,342).  visible(ms3ggt_1,342). happensAt(walking(ms3ggt_1),342). orientation(ms3ggt_2,13,342). coords(ms3ggt_2,209,254,342).  visible(ms3ggt_2,342). happensAt(walking(ms3ggt_2),342).
orientation(ms3ggt_1,156,343). coords(ms3ggt_1,105,166,343).  visible(ms3ggt_1,343). happensAt(walking(ms3ggt_1),343). orientation(ms3ggt_2,13,343). coords(ms3ggt_2,205,253,343).  visible(ms3ggt_2,343). happensAt(walking(ms3ggt_2),343).
orientation(ms3ggt_1,156,344). coords(ms3ggt_1,106,166,344).  visible(ms3ggt_1,344). happensAt(walking(ms3ggt_1),344). orientation(ms3ggt_2,13,344). coords(ms3ggt_2,204,250,344).  visible(ms3ggt_2,344). happensAt(walking(ms3ggt_2),344).
#show holdsAt/2.