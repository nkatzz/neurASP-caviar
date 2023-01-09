
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

	orientation(fomdgt3_1,150,324). coords(fomdgt3_1,54,100,324).  visible(fomdgt3_1,324). happensAt(inactive(fomdgt3_1),324). orientation(fomdgt3_2,60,324). coords(fomdgt3_2,248,98,324).  visible(fomdgt3_2,324). happensAt(walking(fomdgt3_2),324).
orientation(fomdgt3_1,150,325). coords(fomdgt3_1,54,100,325).  visible(fomdgt3_1,325). happensAt(inactive(fomdgt3_1),325). orientation(fomdgt3_2,60,325). coords(fomdgt3_2,248,97,325).  visible(fomdgt3_2,325). happensAt(walking(fomdgt3_2),325).
orientation(fomdgt3_1,150,326). coords(fomdgt3_1,54,100,326).  visible(fomdgt3_1,326). happensAt(inactive(fomdgt3_1),326). orientation(fomdgt3_2,60,326). coords(fomdgt3_2,251,94,326).  visible(fomdgt3_2,326). happensAt(walking(fomdgt3_2),326).
orientation(fomdgt3_1,150,327). coords(fomdgt3_1,54,100,327).  visible(fomdgt3_1,327). happensAt(inactive(fomdgt3_1),327). orientation(fomdgt3_2,60,327). coords(fomdgt3_2,253,91,327).  visible(fomdgt3_2,327). happensAt(walking(fomdgt3_2),327).
orientation(fomdgt3_1,150,328). coords(fomdgt3_1,54,100,328).  visible(fomdgt3_1,328). happensAt(inactive(fomdgt3_1),328). orientation(fomdgt3_2,60,328). coords(fomdgt3_2,252,89,328).  visible(fomdgt3_2,328). happensAt(walking(fomdgt3_2),328).
orientation(fomdgt3_1,150,329). coords(fomdgt3_1,54,100,329).  visible(fomdgt3_1,329). happensAt(inactive(fomdgt3_1),329). orientation(fomdgt3_2,30,329). coords(fomdgt3_2,252,89,329).  visible(fomdgt3_2,329). happensAt(walking(fomdgt3_2),329).
orientation(fomdgt3_1,150,330). coords(fomdgt3_1,54,100,330).  visible(fomdgt3_1,330). happensAt(inactive(fomdgt3_1),330). orientation(fomdgt3_2,30,330). coords(fomdgt3_2,254,88,330).  visible(fomdgt3_2,330). happensAt(walking(fomdgt3_2),330).
orientation(fomdgt3_1,150,331). coords(fomdgt3_1,54,100,331).  visible(fomdgt3_1,331). happensAt(inactive(fomdgt3_1),331). orientation(fomdgt3_2,30,331). coords(fomdgt3_2,254,88,331).  visible(fomdgt3_2,331). happensAt(walking(fomdgt3_2),331).
orientation(fomdgt3_1,150,332). coords(fomdgt3_1,54,100,332).  visible(fomdgt3_1,332). happensAt(inactive(fomdgt3_1),332). orientation(fomdgt3_2,30,332). coords(fomdgt3_2,257,87,332).  visible(fomdgt3_2,332). happensAt(walking(fomdgt3_2),332).
orientation(fomdgt3_1,150,333). coords(fomdgt3_1,54,100,333).  visible(fomdgt3_1,333). happensAt(inactive(fomdgt3_1),333). orientation(fomdgt3_2,30,333). coords(fomdgt3_2,257,86,333).  visible(fomdgt3_2,333). happensAt(walking(fomdgt3_2),333).
orientation(fomdgt3_1,150,334). coords(fomdgt3_1,54,100,334).  visible(fomdgt3_1,334). happensAt(inactive(fomdgt3_1),334). orientation(fomdgt3_2,30,334). coords(fomdgt3_2,257,86,334).  visible(fomdgt3_2,334). happensAt(walking(fomdgt3_2),334).
orientation(fomdgt3_1,150,335). coords(fomdgt3_1,54,100,335).  visible(fomdgt3_1,335). happensAt(inactive(fomdgt3_1),335). orientation(fomdgt3_2,30,335). coords(fomdgt3_2,261,83,335).  visible(fomdgt3_2,335). happensAt(walking(fomdgt3_2),335).
orientation(fomdgt3_1,150,336). coords(fomdgt3_1,54,100,336).  visible(fomdgt3_1,336). happensAt(inactive(fomdgt3_1),336). orientation(fomdgt3_2,30,336). coords(fomdgt3_2,262,81,336).  visible(fomdgt3_2,336). happensAt(walking(fomdgt3_2),336).
orientation(fomdgt3_1,150,337). coords(fomdgt3_1,54,100,337).  visible(fomdgt3_1,337). happensAt(inactive(fomdgt3_1),337). orientation(fomdgt3_2,30,337). coords(fomdgt3_2,263,81,337).  visible(fomdgt3_2,337). happensAt(walking(fomdgt3_2),337).
orientation(fomdgt3_1,150,338). coords(fomdgt3_1,54,100,338).  visible(fomdgt3_1,338). happensAt(inactive(fomdgt3_1),338). orientation(fomdgt3_2,30,338). coords(fomdgt3_2,263,81,338).  visible(fomdgt3_2,338). happensAt(walking(fomdgt3_2),338).
orientation(fomdgt3_1,150,339). coords(fomdgt3_1,54,100,339).  visible(fomdgt3_1,339). happensAt(inactive(fomdgt3_1),339). orientation(fomdgt3_2,30,339). coords(fomdgt3_2,263,81,339).  visible(fomdgt3_2,339). happensAt(walking(fomdgt3_2),339).
orientation(fomdgt3_1,150,340). coords(fomdgt3_1,54,100,340).  visible(fomdgt3_1,340). happensAt(inactive(fomdgt3_1),340). orientation(fomdgt3_2,30,340). coords(fomdgt3_2,263,81,340).  visible(fomdgt3_2,340). happensAt(walking(fomdgt3_2),340).
orientation(fomdgt3_1,150,341). coords(fomdgt3_1,54,100,341).  visible(fomdgt3_1,341). happensAt(inactive(fomdgt3_1),341). orientation(fomdgt3_2,30,341). coords(fomdgt3_2,263,81,341).  visible(fomdgt3_2,341). happensAt(walking(fomdgt3_2),341).
orientation(fomdgt3_1,150,342). coords(fomdgt3_1,54,100,342).  visible(fomdgt3_1,342). happensAt(inactive(fomdgt3_1),342). orientation(fomdgt3_2,30,342). coords(fomdgt3_2,274,81,342).  visible(fomdgt3_2,342). happensAt(walking(fomdgt3_2),342).
orientation(fomdgt3_1,150,343). coords(fomdgt3_1,54,100,343).  visible(fomdgt3_1,343). happensAt(inactive(fomdgt3_1),343). orientation(fomdgt3_2,30,343). coords(fomdgt3_2,274,81,343).  visible(fomdgt3_2,343). happensAt(walking(fomdgt3_2),343).
orientation(fomdgt3_1,150,344). coords(fomdgt3_1,54,100,344).  visible(fomdgt3_1,344). happensAt(inactive(fomdgt3_1),344). orientation(fomdgt3_2,30,344). coords(fomdgt3_2,274,81,344).  visible(fomdgt3_2,344). happensAt(walking(fomdgt3_2),344).
orientation(fomdgt3_1,150,345). coords(fomdgt3_1,54,100,345).  visible(fomdgt3_1,345). happensAt(inactive(fomdgt3_1),345). orientation(fomdgt3_2,30,345). coords(fomdgt3_2,274,81,345).  visible(fomdgt3_2,345). happensAt(walking(fomdgt3_2),345).
orientation(fomdgt3_1,150,346). coords(fomdgt3_1,54,100,346).  visible(fomdgt3_1,346). happensAt(inactive(fomdgt3_1),346). orientation(fomdgt3_2,30,346). coords(fomdgt3_2,274,80,346).  visible(fomdgt3_2,346). happensAt(walking(fomdgt3_2),346).
orientation(fomdgt3_1,150,347). coords(fomdgt3_1,54,100,347).  visible(fomdgt3_1,347). happensAt(inactive(fomdgt3_1),347). orientation(fomdgt3_2,30,347). coords(fomdgt3_2,273,80,347).  visible(fomdgt3_2,347). happensAt(walking(fomdgt3_2),347).
orientation(fomdgt3_1,150,348). coords(fomdgt3_1,54,100,348).  visible(fomdgt3_1,348). happensAt(inactive(fomdgt3_1),348). orientation(fomdgt3_2,30,348). coords(fomdgt3_2,273,80,348).  visible(fomdgt3_2,348). happensAt(walking(fomdgt3_2),348).
#show holdsAt/2.