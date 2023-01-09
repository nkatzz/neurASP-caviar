
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

	orientation(mws1gt_1,155,315). coords(mws1gt_1,78,102,315).  visible(mws1gt_1,315). happensAt(walking(mws1gt_1),315). orientation(mws1gt_7,14,315). coords(mws1gt_7,33,206,315).  visible(mws1gt_7,315). happensAt(walking(mws1gt_7),315).
orientation(mws1gt_1,155,316). coords(mws1gt_1,78,101,316).  visible(mws1gt_1,316). happensAt(walking(mws1gt_1),316). orientation(mws1gt_7,14,316). coords(mws1gt_7,33,206,316).  visible(mws1gt_7,316). happensAt(walking(mws1gt_7),316).
orientation(mws1gt_1,155,317). coords(mws1gt_1,76,101,317).  visible(mws1gt_1,317). happensAt(walking(mws1gt_1),317). orientation(mws1gt_7,14,317). coords(mws1gt_7,34,205,317).  visible(mws1gt_7,317). happensAt(walking(mws1gt_7),317).
orientation(mws1gt_1,155,318). coords(mws1gt_1,76,100,318).  visible(mws1gt_1,318). happensAt(walking(mws1gt_1),318). orientation(mws1gt_7,14,318). coords(mws1gt_7,36,205,318).  visible(mws1gt_7,318). happensAt(walking(mws1gt_7),318).
orientation(mws1gt_1,155,319). coords(mws1gt_1,76,100,319).  visible(mws1gt_1,319). happensAt(walking(mws1gt_1),319). orientation(mws1gt_7,14,319). coords(mws1gt_7,36,204,319).  visible(mws1gt_7,319). happensAt(walking(mws1gt_7),319).
orientation(mws1gt_1,155,320). coords(mws1gt_1,76,100,320).  visible(mws1gt_1,320). happensAt(walking(mws1gt_1),320). orientation(mws1gt_7,14,320). coords(mws1gt_7,36,204,320).  visible(mws1gt_7,320). happensAt(walking(mws1gt_7),320).
orientation(mws1gt_1,155,321). coords(mws1gt_1,76,100,321).  visible(mws1gt_1,321). happensAt(walking(mws1gt_1),321). orientation(mws1gt_7,14,321). coords(mws1gt_7,36,204,321).  visible(mws1gt_7,321). happensAt(walking(mws1gt_7),321).
orientation(mws1gt_1,155,322). coords(mws1gt_1,76,100,322).  visible(mws1gt_1,322). happensAt(walking(mws1gt_1),322). orientation(mws1gt_7,14,322). coords(mws1gt_7,37,202,322).  visible(mws1gt_7,322). happensAt(walking(mws1gt_7),322).
orientation(mws1gt_1,155,323). coords(mws1gt_1,76,100,323).  visible(mws1gt_1,323). happensAt(walking(mws1gt_1),323). orientation(mws1gt_7,14,323). coords(mws1gt_7,37,201,323).  visible(mws1gt_7,323). happensAt(walking(mws1gt_7),323).
orientation(mws1gt_1,155,324). coords(mws1gt_1,75,100,324).  visible(mws1gt_1,324). happensAt(walking(mws1gt_1),324). orientation(mws1gt_7,0,324). coords(mws1gt_7,38,200,324).  visible(mws1gt_7,324). happensAt(walking(mws1gt_7),324).
orientation(mws1gt_1,155,325). coords(mws1gt_1,75,100,325).  visible(mws1gt_1,325). happensAt(walking(mws1gt_1),325). orientation(mws1gt_7,0,325). coords(mws1gt_7,38,199,325).  visible(mws1gt_7,325). happensAt(walking(mws1gt_7),325).
orientation(mws1gt_1,155,326). coords(mws1gt_1,75,100,326).  visible(mws1gt_1,326). happensAt(walking(mws1gt_1),326). orientation(mws1gt_7,170,326). coords(mws1gt_7,38,199,326).  visible(mws1gt_7,326). happensAt(walking(mws1gt_7),326).
orientation(mws1gt_1,155,327). coords(mws1gt_1,75,100,327).  visible(mws1gt_1,327). happensAt(walking(mws1gt_1),327). orientation(mws1gt_7,162,327). coords(mws1gt_7,38,199,327).  visible(mws1gt_7,327). happensAt(walking(mws1gt_7),327).
orientation(mws1gt_1,155,328). coords(mws1gt_1,74,100,328).  visible(mws1gt_1,328). happensAt(walking(mws1gt_1),328). orientation(mws1gt_7,167,328). coords(mws1gt_7,38,197,328).  visible(mws1gt_7,328). happensAt(walking(mws1gt_7),328).
orientation(mws1gt_1,155,329). coords(mws1gt_1,74,99,329).  visible(mws1gt_1,329). happensAt(walking(mws1gt_1),329). orientation(mws1gt_7,167,329). coords(mws1gt_7,39,197,329).  visible(mws1gt_7,329). happensAt(walking(mws1gt_7),329).
orientation(mws1gt_1,155,330). coords(mws1gt_1,73,99,330).  visible(mws1gt_1,330). happensAt(walking(mws1gt_1),330). orientation(mws1gt_7,167,330). coords(mws1gt_7,39,197,330).  visible(mws1gt_7,330). happensAt(walking(mws1gt_7),330).
orientation(mws1gt_1,155,331). coords(mws1gt_1,72,99,331).  visible(mws1gt_1,331). happensAt(walking(mws1gt_1),331). orientation(mws1gt_7,167,331). coords(mws1gt_7,40,197,331).  visible(mws1gt_7,331). happensAt(walking(mws1gt_7),331).
orientation(mws1gt_1,155,332). coords(mws1gt_1,71,99,332).  visible(mws1gt_1,332). happensAt(walking(mws1gt_1),332). orientation(mws1gt_7,167,332). coords(mws1gt_7,40,196,332).  visible(mws1gt_7,332). happensAt(walking(mws1gt_7),332).
orientation(mws1gt_1,155,333). coords(mws1gt_1,71,99,333).  visible(mws1gt_1,333). happensAt(walking(mws1gt_1),333). orientation(mws1gt_7,167,333). coords(mws1gt_7,41,196,333).  visible(mws1gt_7,333). happensAt(walking(mws1gt_7),333).
orientation(mws1gt_1,155,334). coords(mws1gt_1,71,99,334).  visible(mws1gt_1,334). happensAt(walking(mws1gt_1),334). orientation(mws1gt_7,167,334). coords(mws1gt_7,41,196,334).  visible(mws1gt_7,334). happensAt(walking(mws1gt_7),334).
orientation(mws1gt_1,155,335). coords(mws1gt_1,71,99,335).  visible(mws1gt_1,335). happensAt(walking(mws1gt_1),335). orientation(mws1gt_7,167,335). coords(mws1gt_7,41,195,335).  visible(mws1gt_7,335). happensAt(walking(mws1gt_7),335).
orientation(mws1gt_1,155,336). coords(mws1gt_1,71,98,336).  visible(mws1gt_1,336). happensAt(walking(mws1gt_1),336). orientation(mws1gt_7,167,336). coords(mws1gt_7,41,195,336).  visible(mws1gt_7,336). happensAt(walking(mws1gt_7),336).
orientation(mws1gt_1,155,337). coords(mws1gt_1,71,98,337).  visible(mws1gt_1,337). happensAt(walking(mws1gt_1),337). orientation(mws1gt_7,167,337). coords(mws1gt_7,41,194,337).  visible(mws1gt_7,337). happensAt(walking(mws1gt_7),337).
orientation(mws1gt_1,155,338). coords(mws1gt_1,71,98,338).  visible(mws1gt_1,338). happensAt(walking(mws1gt_1),338). orientation(mws1gt_7,167,338). coords(mws1gt_7,41,193,338).  visible(mws1gt_7,338). happensAt(walking(mws1gt_7),338).
orientation(mws1gt_1,155,339). coords(mws1gt_1,71,97,339).  visible(mws1gt_1,339). happensAt(walking(mws1gt_1),339). orientation(mws1gt_7,167,339). coords(mws1gt_7,41,193,339).  visible(mws1gt_7,339). happensAt(walking(mws1gt_7),339).
#show holdsAt/2.