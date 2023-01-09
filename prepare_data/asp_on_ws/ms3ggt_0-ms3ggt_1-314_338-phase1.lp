
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

	orientation(ms3ggt_0,158,314). coords(ms3ggt_0,86,144,314).  visible(ms3ggt_0,314). happensAt(walking(ms3ggt_0),314). orientation(ms3ggt_1,160,314). coords(ms3ggt_1,84,166,314).  visible(ms3ggt_1,314). happensAt(walking(ms3ggt_1),314).
orientation(ms3ggt_0,158,315). coords(ms3ggt_0,87,144,315).  visible(ms3ggt_0,315). happensAt(walking(ms3ggt_0),315). orientation(ms3ggt_1,160,315). coords(ms3ggt_1,85,166,315).  visible(ms3ggt_1,315). happensAt(walking(ms3ggt_1),315).
orientation(ms3ggt_0,153,316). coords(ms3ggt_0,90,144,316).  visible(ms3ggt_0,316). happensAt(walking(ms3ggt_0),316). orientation(ms3ggt_1,160,316). coords(ms3ggt_1,85,166,316).  visible(ms3ggt_1,316). happensAt(walking(ms3ggt_1),316).
orientation(ms3ggt_0,153,317). coords(ms3ggt_0,91,145,317).  visible(ms3ggt_0,317). happensAt(walking(ms3ggt_0),317). orientation(ms3ggt_1,160,317). coords(ms3ggt_1,86,166,317).  visible(ms3ggt_1,317). happensAt(walking(ms3ggt_1),317).
orientation(ms3ggt_0,153,318). coords(ms3ggt_0,92,145,318).  visible(ms3ggt_0,318). happensAt(walking(ms3ggt_0),318). orientation(ms3ggt_1,160,318). coords(ms3ggt_1,88,166,318).  visible(ms3ggt_1,318). happensAt(walking(ms3ggt_1),318).
orientation(ms3ggt_0,153,319). coords(ms3ggt_0,93,145,319).  visible(ms3ggt_0,319). happensAt(walking(ms3ggt_0),319). orientation(ms3ggt_1,162,319). coords(ms3ggt_1,90,166,319).  visible(ms3ggt_1,319). happensAt(walking(ms3ggt_1),319).
orientation(ms3ggt_0,159,320). coords(ms3ggt_0,94,145,320).  visible(ms3ggt_0,320). happensAt(walking(ms3ggt_0),320). orientation(ms3ggt_1,162,320). coords(ms3ggt_1,90,166,320).  visible(ms3ggt_1,320). happensAt(walking(ms3ggt_1),320).
orientation(ms3ggt_0,154,321). coords(ms3ggt_0,94,145,321).  visible(ms3ggt_0,321). happensAt(walking(ms3ggt_0),321). orientation(ms3ggt_1,162,321). coords(ms3ggt_1,91,166,321).  visible(ms3ggt_1,321). happensAt(walking(ms3ggt_1),321).
orientation(ms3ggt_0,154,322). coords(ms3ggt_0,95,145,322).  visible(ms3ggt_0,322). happensAt(walking(ms3ggt_0),322). orientation(ms3ggt_1,162,322). coords(ms3ggt_1,92,166,322).  visible(ms3ggt_1,322). happensAt(walking(ms3ggt_1),322).
orientation(ms3ggt_0,154,323). coords(ms3ggt_0,95,145,323).  visible(ms3ggt_0,323). happensAt(walking(ms3ggt_0),323). orientation(ms3ggt_1,162,323). coords(ms3ggt_1,93,166,323).  visible(ms3ggt_1,323). happensAt(walking(ms3ggt_1),323).
orientation(ms3ggt_0,154,324). coords(ms3ggt_0,95,145,324).  visible(ms3ggt_0,324). happensAt(walking(ms3ggt_0),324). orientation(ms3ggt_1,162,324). coords(ms3ggt_1,93,166,324).  visible(ms3ggt_1,324). happensAt(walking(ms3ggt_1),324).
orientation(ms3ggt_0,155,325). coords(ms3ggt_0,95,145,325).  visible(ms3ggt_0,325). happensAt(walking(ms3ggt_0),325). orientation(ms3ggt_1,162,325). coords(ms3ggt_1,93,167,325).  visible(ms3ggt_1,325). happensAt(walking(ms3ggt_1),325).
orientation(ms3ggt_0,155,326). coords(ms3ggt_0,95,145,326).  visible(ms3ggt_0,326). happensAt(walking(ms3ggt_0),326). orientation(ms3ggt_1,162,326). coords(ms3ggt_1,93,167,326).  visible(ms3ggt_1,326). happensAt(walking(ms3ggt_1),326).
orientation(ms3ggt_0,155,327). coords(ms3ggt_0,96,145,327).  visible(ms3ggt_0,327). happensAt(walking(ms3ggt_0),327). orientation(ms3ggt_1,162,327). coords(ms3ggt_1,93,167,327).  visible(ms3ggt_1,327). happensAt(walking(ms3ggt_1),327).
orientation(ms3ggt_0,155,328). coords(ms3ggt_0,96,145,328).  visible(ms3ggt_0,328). happensAt(walking(ms3ggt_0),328). orientation(ms3ggt_1,162,328). coords(ms3ggt_1,93,167,328).  visible(ms3ggt_1,328). happensAt(walking(ms3ggt_1),328).
orientation(ms3ggt_0,155,329). coords(ms3ggt_0,98,145,329).  visible(ms3ggt_0,329). happensAt(walking(ms3ggt_0),329). orientation(ms3ggt_1,156,329). coords(ms3ggt_1,94,167,329).  visible(ms3ggt_1,329). happensAt(walking(ms3ggt_1),329).
orientation(ms3ggt_0,159,330). coords(ms3ggt_0,100,145,330).  visible(ms3ggt_0,330). happensAt(walking(ms3ggt_0),330). orientation(ms3ggt_1,156,330). coords(ms3ggt_1,95,167,330).  visible(ms3ggt_1,330). happensAt(walking(ms3ggt_1),330).
orientation(ms3ggt_0,159,331). coords(ms3ggt_0,101,145,331).  visible(ms3ggt_0,331). happensAt(walking(ms3ggt_0),331). orientation(ms3ggt_1,156,331). coords(ms3ggt_1,95,167,331).  visible(ms3ggt_1,331). happensAt(walking(ms3ggt_1),331).
orientation(ms3ggt_0,163,332). coords(ms3ggt_0,103,145,332).  visible(ms3ggt_0,332). happensAt(walking(ms3ggt_0),332). orientation(ms3ggt_1,156,332). coords(ms3ggt_1,97,167,332).  visible(ms3ggt_1,332). happensAt(walking(ms3ggt_1),332).
orientation(ms3ggt_0,163,333). coords(ms3ggt_0,104,145,333).  visible(ms3ggt_0,333). happensAt(walking(ms3ggt_0),333). orientation(ms3ggt_1,156,333). coords(ms3ggt_1,98,167,333).  visible(ms3ggt_1,333). happensAt(walking(ms3ggt_1),333).
orientation(ms3ggt_0,163,334). coords(ms3ggt_0,105,145,334).  visible(ms3ggt_0,334). happensAt(walking(ms3ggt_0),334). orientation(ms3ggt_1,156,334). coords(ms3ggt_1,100,167,334).  visible(ms3ggt_1,334). happensAt(walking(ms3ggt_1),334).
orientation(ms3ggt_0,163,335). coords(ms3ggt_0,106,145,335).  visible(ms3ggt_0,335). happensAt(walking(ms3ggt_0),335). orientation(ms3ggt_1,156,335). coords(ms3ggt_1,101,166,335).  visible(ms3ggt_1,335). happensAt(walking(ms3ggt_1),335).
orientation(ms3ggt_0,163,336). coords(ms3ggt_0,106,145,336).  visible(ms3ggt_0,336). happensAt(walking(ms3ggt_0),336). orientation(ms3ggt_1,156,336). coords(ms3ggt_1,103,166,336).  visible(ms3ggt_1,336). happensAt(walking(ms3ggt_1),336).
orientation(ms3ggt_0,163,337). coords(ms3ggt_0,106,145,337).  visible(ms3ggt_0,337). happensAt(walking(ms3ggt_0),337). orientation(ms3ggt_1,156,337). coords(ms3ggt_1,103,166,337).  visible(ms3ggt_1,337). happensAt(walking(ms3ggt_1),337).
orientation(ms3ggt_0,163,338). coords(ms3ggt_0,107,145,338).  visible(ms3ggt_0,338). happensAt(walking(ms3ggt_0),338). orientation(ms3ggt_1,156,338). coords(ms3ggt_1,104,166,338).  visible(ms3ggt_1,338). happensAt(walking(ms3ggt_1),338).
#show holdsAt/2.