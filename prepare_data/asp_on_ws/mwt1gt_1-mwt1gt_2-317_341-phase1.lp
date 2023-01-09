
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

	orientation(mwt1gt_1,160,317). coords(mwt1gt_1,111,155,317).  visible(mwt1gt_1,317). happensAt(walking(mwt1gt_1),317). orientation(mwt1gt_2,140,317). coords(mwt1gt_2,137,116,317).  visible(mwt1gt_2,317). happensAt(walking(mwt1gt_2),317).
orientation(mwt1gt_1,160,318). coords(mwt1gt_1,111,155,318).  visible(mwt1gt_1,318). happensAt(walking(mwt1gt_1),318). orientation(mwt1gt_2,140,318). coords(mwt1gt_2,137,119,318).  visible(mwt1gt_2,318). happensAt(walking(mwt1gt_2),318).
orientation(mwt1gt_1,160,319). coords(mwt1gt_1,114,153,319).  visible(mwt1gt_1,319). happensAt(walking(mwt1gt_1),319). orientation(mwt1gt_2,140,319). coords(mwt1gt_2,137,119,319).  visible(mwt1gt_2,319). happensAt(walking(mwt1gt_2),319).
orientation(mwt1gt_1,160,320). coords(mwt1gt_1,115,153,320).  visible(mwt1gt_1,320). happensAt(walking(mwt1gt_1),320). orientation(mwt1gt_2,140,320). coords(mwt1gt_2,136,120,320).  visible(mwt1gt_2,320). happensAt(walking(mwt1gt_2),320).
orientation(mwt1gt_1,160,321). coords(mwt1gt_1,115,153,321).  visible(mwt1gt_1,321). happensAt(walking(mwt1gt_1),321). orientation(mwt1gt_2,140,321). coords(mwt1gt_2,136,120,321).  visible(mwt1gt_2,321). happensAt(walking(mwt1gt_2),321).
orientation(mwt1gt_1,160,322). coords(mwt1gt_1,115,153,322).  visible(mwt1gt_1,322). happensAt(walking(mwt1gt_1),322). orientation(mwt1gt_2,140,322). coords(mwt1gt_2,133,121,322).  visible(mwt1gt_2,322). happensAt(walking(mwt1gt_2),322).
orientation(mwt1gt_1,160,323). coords(mwt1gt_1,115,153,323).  visible(mwt1gt_1,323). happensAt(walking(mwt1gt_1),323). orientation(mwt1gt_2,140,323). coords(mwt1gt_2,133,122,323).  visible(mwt1gt_2,323). happensAt(walking(mwt1gt_2),323).
orientation(mwt1gt_1,160,324). coords(mwt1gt_1,117,152,324).  visible(mwt1gt_1,324). happensAt(walking(mwt1gt_1),324). orientation(mwt1gt_2,140,324). coords(mwt1gt_2,133,123,324).  visible(mwt1gt_2,324). happensAt(walking(mwt1gt_2),324).
orientation(mwt1gt_1,160,325). coords(mwt1gt_1,118,151,325).  visible(mwt1gt_1,325). happensAt(walking(mwt1gt_1),325). orientation(mwt1gt_2,140,325). coords(mwt1gt_2,133,121,325).  visible(mwt1gt_2,325). happensAt(walking(mwt1gt_2),325).
orientation(mwt1gt_1,160,326). coords(mwt1gt_1,118,151,326).  visible(mwt1gt_1,326). happensAt(walking(mwt1gt_1),326). orientation(mwt1gt_2,140,326). coords(mwt1gt_2,130,123,326).  visible(mwt1gt_2,326). happensAt(walking(mwt1gt_2),326).
orientation(mwt1gt_1,160,327). coords(mwt1gt_1,118,151,327).  visible(mwt1gt_1,327). happensAt(walking(mwt1gt_1),327). orientation(mwt1gt_2,140,327). coords(mwt1gt_2,130,125,327).  visible(mwt1gt_2,327). happensAt(walking(mwt1gt_2),327).
orientation(mwt1gt_1,160,328). coords(mwt1gt_1,118,151,328).  visible(mwt1gt_1,328). happensAt(walking(mwt1gt_1),328). orientation(mwt1gt_2,140,328). coords(mwt1gt_2,130,126,328).  visible(mwt1gt_2,328). happensAt(walking(mwt1gt_2),328).
orientation(mwt1gt_1,160,329). coords(mwt1gt_1,118,151,329).  visible(mwt1gt_1,329). happensAt(walking(mwt1gt_1),329). orientation(mwt1gt_2,140,329). coords(mwt1gt_2,129,126,329).  visible(mwt1gt_2,329). happensAt(walking(mwt1gt_2),329).
orientation(mwt1gt_1,160,330). coords(mwt1gt_1,119,150,330).  visible(mwt1gt_1,330). happensAt(active(mwt1gt_1),330). orientation(mwt1gt_2,140,330). coords(mwt1gt_2,129,126,330).  visible(mwt1gt_2,330). happensAt(walking(mwt1gt_2),330).
orientation(mwt1gt_1,160,331). coords(mwt1gt_1,119,150,331).  visible(mwt1gt_1,331). happensAt(active(mwt1gt_1),331). orientation(mwt1gt_2,140,331). coords(mwt1gt_2,129,126,331).  visible(mwt1gt_2,331). happensAt(walking(mwt1gt_2),331).
orientation(mwt1gt_1,160,332). coords(mwt1gt_1,119,150,332).  visible(mwt1gt_1,332). happensAt(active(mwt1gt_1),332). orientation(mwt1gt_2,140,332). coords(mwt1gt_2,128,126,332).  visible(mwt1gt_2,332). happensAt(active(mwt1gt_2),332).
orientation(mwt1gt_1,160,333). coords(mwt1gt_1,119,150,333).  visible(mwt1gt_1,333). happensAt(active(mwt1gt_1),333). orientation(mwt1gt_2,140,333). coords(mwt1gt_2,128,127,333).  visible(mwt1gt_2,333). happensAt(active(mwt1gt_2),333).
orientation(mwt1gt_1,160,334). coords(mwt1gt_1,119,150,334).  visible(mwt1gt_1,334). happensAt(active(mwt1gt_1),334). orientation(mwt1gt_2,140,334). coords(mwt1gt_2,128,127,334).  visible(mwt1gt_2,334). happensAt(active(mwt1gt_2),334).
orientation(mwt1gt_1,160,335). coords(mwt1gt_1,119,149,335).  visible(mwt1gt_1,335). happensAt(active(mwt1gt_1),335). orientation(mwt1gt_2,140,335). coords(mwt1gt_2,128,127,335).  visible(mwt1gt_2,335). happensAt(active(mwt1gt_2),335).
orientation(mwt1gt_1,160,336). coords(mwt1gt_1,119,149,336).  visible(mwt1gt_1,336). happensAt(active(mwt1gt_1),336). orientation(mwt1gt_2,140,336). coords(mwt1gt_2,127,127,336).  visible(mwt1gt_2,336). happensAt(active(mwt1gt_2),336).
orientation(mwt1gt_1,160,337). coords(mwt1gt_1,119,149,337).  visible(mwt1gt_1,337). happensAt(active(mwt1gt_1),337). orientation(mwt1gt_2,140,337). coords(mwt1gt_2,127,127,337).  visible(mwt1gt_2,337). happensAt(active(mwt1gt_2),337).
orientation(mwt1gt_1,160,338). coords(mwt1gt_1,119,148,338).  visible(mwt1gt_1,338). happensAt(active(mwt1gt_1),338). orientation(mwt1gt_2,140,338). coords(mwt1gt_2,127,127,338).  visible(mwt1gt_2,338). happensAt(active(mwt1gt_2),338).
orientation(mwt1gt_1,160,339). coords(mwt1gt_1,119,148,339).  visible(mwt1gt_1,339). happensAt(active(mwt1gt_1),339). orientation(mwt1gt_2,140,339). coords(mwt1gt_2,127,127,339).  visible(mwt1gt_2,339). happensAt(active(mwt1gt_2),339).
orientation(mwt1gt_1,160,340). coords(mwt1gt_1,118,148,340).  visible(mwt1gt_1,340). happensAt(active(mwt1gt_1),340). orientation(mwt1gt_2,140,340). coords(mwt1gt_2,127,127,340).  visible(mwt1gt_2,340). happensAt(active(mwt1gt_2),340).
orientation(mwt1gt_1,160,341). coords(mwt1gt_1,118,148,341).  visible(mwt1gt_1,341). happensAt(active(mwt1gt_1),341). orientation(mwt1gt_2,140,341). coords(mwt1gt_2,127,127,341).  visible(mwt1gt_2,341). happensAt(active(mwt1gt_2),341).
#show holdsAt/2.