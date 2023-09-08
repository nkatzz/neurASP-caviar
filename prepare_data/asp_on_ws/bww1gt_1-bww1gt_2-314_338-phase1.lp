
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

	orientation(bww1gt_1,7,314). coords(bww1gt_1,105,264,314).  visible(bww1gt_1,314). happensAt(inactive(bww1gt_1),314). orientation(bww1gt_2,167,314). coords(bww1gt_2,63,161,314).  visible(bww1gt_2,314). happensAt(walking(bww1gt_2),314).
orientation(bww1gt_1,4,315). coords(bww1gt_1,105,264,315).  visible(bww1gt_1,315). happensAt(inactive(bww1gt_1),315). orientation(bww1gt_2,167,315). coords(bww1gt_2,63,161,315).  visible(bww1gt_2,315). happensAt(walking(bww1gt_2),315).
orientation(bww1gt_1,6,316). coords(bww1gt_1,105,264,316).  visible(bww1gt_1,316). happensAt(inactive(bww1gt_1),316). orientation(bww1gt_2,167,316). coords(bww1gt_2,63,161,316).  visible(bww1gt_2,316). happensAt(walking(bww1gt_2),316).
orientation(bww1gt_1,4,317). coords(bww1gt_1,105,264,317).  visible(bww1gt_1,317). happensAt(inactive(bww1gt_1),317). orientation(bww1gt_2,167,317). coords(bww1gt_2,63,161,317).  visible(bww1gt_2,317). happensAt(walking(bww1gt_2),317).
orientation(bww1gt_1,4,318). coords(bww1gt_1,105,264,318).  visible(bww1gt_1,318). happensAt(inactive(bww1gt_1),318). orientation(bww1gt_2,167,318). coords(bww1gt_2,63,161,318).  visible(bww1gt_2,318). happensAt(walking(bww1gt_2),318).
orientation(bww1gt_1,2,319). coords(bww1gt_1,105,264,319).  visible(bww1gt_1,319). happensAt(inactive(bww1gt_1),319). orientation(bww1gt_2,167,319). coords(bww1gt_2,64,161,319).  visible(bww1gt_2,319). happensAt(walking(bww1gt_2),319).
orientation(bww1gt_1,4,320). coords(bww1gt_1,105,264,320).  visible(bww1gt_1,320). happensAt(inactive(bww1gt_1),320). orientation(bww1gt_2,167,320). coords(bww1gt_2,65,161,320).  visible(bww1gt_2,320). happensAt(walking(bww1gt_2),320).
orientation(bww1gt_1,4,321). coords(bww1gt_1,105,264,321).  visible(bww1gt_1,321). happensAt(inactive(bww1gt_1),321). orientation(bww1gt_2,167,321). coords(bww1gt_2,65,161,321).  visible(bww1gt_2,321). happensAt(walking(bww1gt_2),321).
orientation(bww1gt_1,5,322). coords(bww1gt_1,105,264,322).  visible(bww1gt_1,322). happensAt(inactive(bww1gt_1),322). orientation(bww1gt_2,167,322). coords(bww1gt_2,66,161,322).  visible(bww1gt_2,322). happensAt(walking(bww1gt_2),322).
orientation(bww1gt_1,8,323). coords(bww1gt_1,105,264,323).  visible(bww1gt_1,323). happensAt(inactive(bww1gt_1),323). orientation(bww1gt_2,167,323). coords(bww1gt_2,67,161,323).  visible(bww1gt_2,323). happensAt(walking(bww1gt_2),323).
orientation(bww1gt_1,8,324). coords(bww1gt_1,105,264,324).  visible(bww1gt_1,324). happensAt(inactive(bww1gt_1),324). orientation(bww1gt_2,167,324). coords(bww1gt_2,68,160,324).  visible(bww1gt_2,324). happensAt(walking(bww1gt_2),324).
orientation(bww1gt_1,7,325). coords(bww1gt_1,105,264,325).  visible(bww1gt_1,325). happensAt(inactive(bww1gt_1),325). orientation(bww1gt_2,167,325). coords(bww1gt_2,68,160,325).  visible(bww1gt_2,325). happensAt(walking(bww1gt_2),325).
orientation(bww1gt_1,179,326). coords(bww1gt_1,105,264,326).  visible(bww1gt_1,326). happensAt(inactive(bww1gt_1),326). orientation(bww1gt_2,167,326). coords(bww1gt_2,69,160,326).  visible(bww1gt_2,326). happensAt(walking(bww1gt_2),326).
orientation(bww1gt_1,9,327). coords(bww1gt_1,105,264,327).  visible(bww1gt_1,327). happensAt(inactive(bww1gt_1),327). orientation(bww1gt_2,167,327). coords(bww1gt_2,69,160,327).  visible(bww1gt_2,327). happensAt(walking(bww1gt_2),327).
orientation(bww1gt_1,8,328). coords(bww1gt_1,105,264,328).  visible(bww1gt_1,328). happensAt(inactive(bww1gt_1),328). orientation(bww1gt_2,167,328). coords(bww1gt_2,70,160,328).  visible(bww1gt_2,328). happensAt(walking(bww1gt_2),328).
orientation(bww1gt_1,7,329). coords(bww1gt_1,105,264,329).  visible(bww1gt_1,329). happensAt(inactive(bww1gt_1),329). orientation(bww1gt_2,167,329). coords(bww1gt_2,70,160,329).  visible(bww1gt_2,329). happensAt(walking(bww1gt_2),329).
orientation(bww1gt_1,8,330). coords(bww1gt_1,105,264,330).  visible(bww1gt_1,330). happensAt(inactive(bww1gt_1),330). orientation(bww1gt_2,167,330). coords(bww1gt_2,70,160,330).  visible(bww1gt_2,330). happensAt(walking(bww1gt_2),330).
orientation(bww1gt_1,5,331). coords(bww1gt_1,105,264,331).  visible(bww1gt_1,331). happensAt(inactive(bww1gt_1),331). orientation(bww1gt_2,167,331). coords(bww1gt_2,70,160,331).  visible(bww1gt_2,331). happensAt(walking(bww1gt_2),331).
orientation(bww1gt_1,4,332). coords(bww1gt_1,105,264,332).  visible(bww1gt_1,332). happensAt(inactive(bww1gt_1),332). orientation(bww1gt_2,167,332). coords(bww1gt_2,70,160,332).  visible(bww1gt_2,332). happensAt(walking(bww1gt_2),332).
orientation(bww1gt_1,4,333). coords(bww1gt_1,105,264,333).  visible(bww1gt_1,333). happensAt(inactive(bww1gt_1),333). orientation(bww1gt_2,167,333). coords(bww1gt_2,70,160,333).  visible(bww1gt_2,333). happensAt(walking(bww1gt_2),333).
orientation(bww1gt_1,6,334). coords(bww1gt_1,105,264,334).  visible(bww1gt_1,334). happensAt(inactive(bww1gt_1),334). orientation(bww1gt_2,167,334). coords(bww1gt_2,70,159,334).  visible(bww1gt_2,334). happensAt(walking(bww1gt_2),334).
orientation(bww1gt_1,6,335). coords(bww1gt_1,105,264,335).  visible(bww1gt_1,335). happensAt(inactive(bww1gt_1),335). orientation(bww1gt_2,167,335). coords(bww1gt_2,71,159,335).  visible(bww1gt_2,335). happensAt(walking(bww1gt_2),335).
orientation(bww1gt_1,9,336). coords(bww1gt_1,105,264,336).  visible(bww1gt_1,336). happensAt(inactive(bww1gt_1),336). orientation(bww1gt_2,167,336). coords(bww1gt_2,72,159,336).  visible(bww1gt_2,336). happensAt(walking(bww1gt_2),336).
orientation(bww1gt_1,5,337). coords(bww1gt_1,105,264,337).  visible(bww1gt_1,337). happensAt(inactive(bww1gt_1),337). orientation(bww1gt_2,167,337). coords(bww1gt_2,73,159,337).  visible(bww1gt_2,337). happensAt(walking(bww1gt_2),337).
orientation(bww1gt_1,7,338). coords(bww1gt_1,105,264,338).  visible(bww1gt_1,338). happensAt(inactive(bww1gt_1),338). orientation(bww1gt_2,167,338). coords(bww1gt_2,75,159,338).  visible(bww1gt_2,338). happensAt(walking(bww1gt_2),338).
#show holdsAt/2.