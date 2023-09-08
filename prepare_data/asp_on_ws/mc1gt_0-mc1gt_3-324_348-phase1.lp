
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

	orientation(mc1gt_0,10,324). coords(mc1gt_0,213,263,324).  visible(mc1gt_0,324). happensAt(walking(mc1gt_0),324). orientation(mc1gt_3,42,324). coords(mc1gt_3,311,236,324).  visible(mc1gt_3,324). happensAt(walking(mc1gt_3),324).
orientation(mc1gt_0,10,325). coords(mc1gt_0,213,266,325).  visible(mc1gt_0,325). happensAt(walking(mc1gt_0),325). orientation(mc1gt_3,61,325). coords(mc1gt_3,311,236,325).  visible(mc1gt_3,325). happensAt(walking(mc1gt_3),325).
orientation(mc1gt_0,10,326). coords(mc1gt_0,213,267,326).  visible(mc1gt_0,326). happensAt(walking(mc1gt_0),326). orientation(mc1gt_3,61,326). coords(mc1gt_3,312,239,326).  visible(mc1gt_3,326). happensAt(walking(mc1gt_3),326).
orientation(mc1gt_0,10,327). coords(mc1gt_0,213,268,327).  visible(mc1gt_0,327). happensAt(walking(mc1gt_0),327). orientation(mc1gt_3,60,327). coords(mc1gt_3,312,241,327).  visible(mc1gt_3,327). happensAt(walking(mc1gt_3),327).
orientation(mc1gt_0,16,328). coords(mc1gt_0,214,269,328).  visible(mc1gt_0,328). happensAt(walking(mc1gt_0),328). orientation(mc1gt_3,53,328). coords(mc1gt_3,312,244,328).  visible(mc1gt_3,328). happensAt(walking(mc1gt_3),328).
orientation(mc1gt_0,19,329). coords(mc1gt_0,214,269,329).  visible(mc1gt_0,329). happensAt(walking(mc1gt_0),329). orientation(mc1gt_3,56,329). coords(mc1gt_3,312,248,329).  visible(mc1gt_3,329). happensAt(walking(mc1gt_3),329).
orientation(mc1gt_0,22,330). coords(mc1gt_0,214,270,330).  visible(mc1gt_0,330). happensAt(walking(mc1gt_0),330). orientation(mc1gt_3,62,330). coords(mc1gt_3,312,252,330).  visible(mc1gt_3,330). happensAt(walking(mc1gt_3),330).
orientation(mc1gt_0,22,331). coords(mc1gt_0,214,272,331).  visible(mc1gt_0,331). happensAt(walking(mc1gt_0),331). orientation(mc1gt_3,50,331). coords(mc1gt_3,313,256,331).  visible(mc1gt_3,331). happensAt(walking(mc1gt_3),331).
orientation(mc1gt_0,22,332). coords(mc1gt_0,214,272,332).  visible(mc1gt_0,332). happensAt(walking(mc1gt_0),332). orientation(mc1gt_3,51,332). coords(mc1gt_3,313,261,332).  visible(mc1gt_3,332). happensAt(walking(mc1gt_3),332).
orientation(mc1gt_0,22,333). coords(mc1gt_0,214,272,333).  visible(mc1gt_0,333). happensAt(walking(mc1gt_0),333). orientation(mc1gt_3,50,333). coords(mc1gt_3,313,264,333).  visible(mc1gt_3,333). happensAt(walking(mc1gt_3),333).
orientation(mc1gt_0,22,334). coords(mc1gt_0,214,272,334).  visible(mc1gt_0,334). happensAt(walking(mc1gt_0),334). orientation(mc1gt_3,53,334). coords(mc1gt_3,313,265,334).  visible(mc1gt_3,334). happensAt(walking(mc1gt_3),334).
orientation(mc1gt_0,22,335). coords(mc1gt_0,214,272,335).  visible(mc1gt_0,335). happensAt(walking(mc1gt_0),335). orientation(mc1gt_3,58,335). coords(mc1gt_3,313,266,335).  visible(mc1gt_3,335). happensAt(walking(mc1gt_3),335).
orientation(mc1gt_0,22,336). coords(mc1gt_0,214,273,336).  visible(mc1gt_0,336). happensAt(walking(mc1gt_0),336). orientation(mc1gt_3,59,336). coords(mc1gt_3,312,266,336).  visible(mc1gt_3,336). happensAt(walking(mc1gt_3),336).
orientation(mc1gt_0,22,337). coords(mc1gt_0,214,274,337).  visible(mc1gt_0,337). happensAt(walking(mc1gt_0),337). orientation(mc1gt_3,59,337). coords(mc1gt_3,312,266,337).  visible(mc1gt_3,337). happensAt(walking(mc1gt_3),337).
orientation(mc1gt_0,22,338). coords(mc1gt_0,216,275,338).  visible(mc1gt_0,338). happensAt(walking(mc1gt_0),338). orientation(mc1gt_3,61,338). coords(mc1gt_3,311,266,338).  visible(mc1gt_3,338). happensAt(walking(mc1gt_3),338).
orientation(mc1gt_0,22,339). coords(mc1gt_0,216,276,339).  visible(mc1gt_0,339). happensAt(walking(mc1gt_0),339). orientation(mc1gt_3,60,339). coords(mc1gt_3,311,266,339).  visible(mc1gt_3,339). happensAt(walking(mc1gt_3),339).
orientation(mc1gt_0,22,340). coords(mc1gt_0,217,278,340).  visible(mc1gt_0,340). happensAt(walking(mc1gt_0),340). orientation(mc1gt_3,60,340). coords(mc1gt_3,311,267,340).  visible(mc1gt_3,340). happensAt(walking(mc1gt_3),340).
orientation(mc1gt_0,22,341). coords(mc1gt_0,218,280,341).  visible(mc1gt_0,341). happensAt(walking(mc1gt_0),341). orientation(mc1gt_3,50,341). coords(mc1gt_3,311,270,341).  visible(mc1gt_3,341). happensAt(walking(mc1gt_3),341).
orientation(mc1gt_0,22,342). coords(mc1gt_0,219,281,342).  visible(mc1gt_0,342). happensAt(walking(mc1gt_0),342). orientation(mc1gt_3,70,342). coords(mc1gt_3,311,272,342).  visible(mc1gt_3,342). happensAt(walking(mc1gt_3),342).
orientation(mc1gt_0,22,343). coords(mc1gt_0,221,281,343).  visible(mc1gt_0,343). happensAt(walking(mc1gt_0),343). orientation(mc1gt_3,72,343). coords(mc1gt_3,311,275,343).  visible(mc1gt_3,343). happensAt(walking(mc1gt_3),343).
orientation(mc1gt_0,22,344). coords(mc1gt_0,221,281,344).  visible(mc1gt_0,344). happensAt(walking(mc1gt_0),344). orientation(mc1gt_3,72,344). coords(mc1gt_3,312,278,344).  visible(mc1gt_3,344). happensAt(walking(mc1gt_3),344).
orientation(mc1gt_0,28,345). coords(mc1gt_0,220,281,345).  visible(mc1gt_0,345). happensAt(walking(mc1gt_0),345). orientation(mc1gt_3,64,345). coords(mc1gt_3,314,281,345).  visible(mc1gt_3,345). happensAt(walking(mc1gt_3),345).
orientation(mc1gt_0,31,346). coords(mc1gt_0,221,281,346).  visible(mc1gt_0,346). happensAt(walking(mc1gt_0),346). orientation(mc1gt_3,65,346). coords(mc1gt_3,315,282,346).  visible(mc1gt_3,346). happensAt(walking(mc1gt_3),346).
orientation(mc1gt_0,33,347). coords(mc1gt_0,221,281,347).  visible(mc1gt_0,347). happensAt(walking(mc1gt_0),347). orientation(mc1gt_3,29,347). coords(mc1gt_3,316,282,347).  visible(mc1gt_3,347). happensAt(walking(mc1gt_3),347).
orientation(mc1gt_0,39,348). coords(mc1gt_0,223,281,348).  visible(mc1gt_0,348). happensAt(walking(mc1gt_0),348). orientation(mc1gt_3,176,348). coords(mc1gt_3,316,283,348).  visible(mc1gt_3,348). happensAt(walking(mc1gt_3),348).
#show holdsAt/2.