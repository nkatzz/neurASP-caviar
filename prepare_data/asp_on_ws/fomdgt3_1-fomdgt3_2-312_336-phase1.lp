
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

	orientation(fomdgt3_1,150,312). coords(fomdgt3_1,54,100,312).  visible(fomdgt3_1,312). happensAt(inactive(fomdgt3_1),312). orientation(fomdgt3_2,120,312). coords(fomdgt3_2,240,116,312).  visible(fomdgt3_2,312). happensAt(walking(fomdgt3_2),312).
orientation(fomdgt3_1,150,313). coords(fomdgt3_1,54,100,313).  visible(fomdgt3_1,313). happensAt(inactive(fomdgt3_1),313). orientation(fomdgt3_2,120,313). coords(fomdgt3_2,241,115,313).  visible(fomdgt3_2,313). happensAt(walking(fomdgt3_2),313).
orientation(fomdgt3_1,150,314). coords(fomdgt3_1,54,100,314).  visible(fomdgt3_1,314). happensAt(inactive(fomdgt3_1),314). orientation(fomdgt3_2,120,314). coords(fomdgt3_2,242,114,314).  visible(fomdgt3_2,314). happensAt(walking(fomdgt3_2),314).
orientation(fomdgt3_1,150,315). coords(fomdgt3_1,54,100,315).  visible(fomdgt3_1,315). happensAt(inactive(fomdgt3_1),315). orientation(fomdgt3_2,120,315). coords(fomdgt3_2,243,113,315).  visible(fomdgt3_2,315). happensAt(walking(fomdgt3_2),315).
orientation(fomdgt3_1,150,316). coords(fomdgt3_1,54,100,316).  visible(fomdgt3_1,316). happensAt(inactive(fomdgt3_1),316). orientation(fomdgt3_2,120,316). coords(fomdgt3_2,243,111,316).  visible(fomdgt3_2,316). happensAt(walking(fomdgt3_2),316).
orientation(fomdgt3_1,150,317). coords(fomdgt3_1,54,100,317).  visible(fomdgt3_1,317). happensAt(inactive(fomdgt3_1),317). orientation(fomdgt3_2,80,317). coords(fomdgt3_2,243,110,317).  visible(fomdgt3_2,317). happensAt(walking(fomdgt3_2),317).
orientation(fomdgt3_1,150,318). coords(fomdgt3_1,54,100,318).  visible(fomdgt3_1,318). happensAt(inactive(fomdgt3_1),318). orientation(fomdgt3_2,80,318). coords(fomdgt3_2,244,110,318).  visible(fomdgt3_2,318). happensAt(walking(fomdgt3_2),318).
orientation(fomdgt3_1,150,319). coords(fomdgt3_1,54,100,319).  visible(fomdgt3_1,319). happensAt(inactive(fomdgt3_1),319). orientation(fomdgt3_2,80,319). coords(fomdgt3_2,249,107,319).  visible(fomdgt3_2,319). happensAt(walking(fomdgt3_2),319).
orientation(fomdgt3_1,150,320). coords(fomdgt3_1,54,100,320).  visible(fomdgt3_1,320). happensAt(inactive(fomdgt3_1),320). orientation(fomdgt3_2,80,320). coords(fomdgt3_2,249,104,320).  visible(fomdgt3_2,320). happensAt(walking(fomdgt3_2),320).
orientation(fomdgt3_1,150,321). coords(fomdgt3_1,54,100,321).  visible(fomdgt3_1,321). happensAt(inactive(fomdgt3_1),321). orientation(fomdgt3_2,80,321). coords(fomdgt3_2,250,104,321).  visible(fomdgt3_2,321). happensAt(walking(fomdgt3_2),321).
orientation(fomdgt3_1,150,322). coords(fomdgt3_1,54,100,322).  visible(fomdgt3_1,322). happensAt(inactive(fomdgt3_1),322). orientation(fomdgt3_2,70,322). coords(fomdgt3_2,248,102,322).  visible(fomdgt3_2,322). happensAt(walking(fomdgt3_2),322).
orientation(fomdgt3_1,150,323). coords(fomdgt3_1,54,100,323).  visible(fomdgt3_1,323). happensAt(inactive(fomdgt3_1),323). orientation(fomdgt3_2,70,323). coords(fomdgt3_2,248,102,323).  visible(fomdgt3_2,323). happensAt(walking(fomdgt3_2),323).
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
#show holdsAt/2.