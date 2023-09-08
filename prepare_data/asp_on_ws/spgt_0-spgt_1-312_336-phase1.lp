
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

	orientation(spgt_0,149,312). coords(spgt_0,184,133,312).  visible(spgt_0,312). happensAt(walking(spgt_0),312). orientation(spgt_1,133,312). coords(spgt_1,145,42,312).  visible(spgt_1,312). happensAt(walking(spgt_1),312).
orientation(spgt_0,149,313). coords(spgt_0,185,134,313).  visible(spgt_0,313). happensAt(walking(spgt_0),313). orientation(spgt_1,133,313). coords(spgt_1,145,41,313).  visible(spgt_1,313). happensAt(walking(spgt_1),313).
orientation(spgt_0,149,314). coords(spgt_0,186,135,314).  visible(spgt_0,314). happensAt(walking(spgt_0),314). orientation(spgt_1,133,314). coords(spgt_1,146,41,314).  visible(spgt_1,314). happensAt(walking(spgt_1),314).
orientation(spgt_0,149,315). coords(spgt_0,186,135,315).  visible(spgt_0,315). happensAt(walking(spgt_0),315). orientation(spgt_1,133,315). coords(spgt_1,147,41,315).  visible(spgt_1,315). happensAt(walking(spgt_1),315).
orientation(spgt_0,149,316). coords(spgt_0,187,136,316).  visible(spgt_0,316). happensAt(walking(spgt_0),316). orientation(spgt_1,133,316). coords(spgt_1,148,40,316).  visible(spgt_1,316). happensAt(walking(spgt_1),316).
orientation(spgt_0,149,317). coords(spgt_0,187,137,317).  visible(spgt_0,317). happensAt(walking(spgt_0),317). orientation(spgt_1,133,317). coords(spgt_1,148,40,317).  visible(spgt_1,317). happensAt(walking(spgt_1),317).
orientation(spgt_0,149,318). coords(spgt_0,187,140,318).  visible(spgt_0,318). happensAt(walking(spgt_0),318). orientation(spgt_1,133,318). coords(spgt_1,150,40,318).  visible(spgt_1,318). happensAt(walking(spgt_1),318).
orientation(spgt_0,149,319). coords(spgt_0,188,144,319).  visible(spgt_0,319). happensAt(walking(spgt_0),319). orientation(spgt_1,133,319). coords(spgt_1,150,39,319).  visible(spgt_1,319). happensAt(walking(spgt_1),319).
orientation(spgt_0,149,320). coords(spgt_0,188,146,320).  visible(spgt_0,320). happensAt(walking(spgt_0),320). orientation(spgt_1,133,320). coords(spgt_1,150,37,320).  visible(spgt_1,320). happensAt(walking(spgt_1),320).
orientation(spgt_0,149,321). coords(spgt_0,190,150,321).  visible(spgt_0,321). happensAt(walking(spgt_0),321). orientation(spgt_1,133,321). coords(spgt_1,151,37,321).  visible(spgt_1,321). happensAt(walking(spgt_1),321).
orientation(spgt_0,149,322). coords(spgt_0,191,151,322).  visible(spgt_0,322). happensAt(walking(spgt_0),322). orientation(spgt_1,133,322). coords(spgt_1,151,37,322).  visible(spgt_1,322). happensAt(walking(spgt_1),322).
orientation(spgt_0,149,323). coords(spgt_0,193,153,323).  visible(spgt_0,323). happensAt(walking(spgt_0),323). orientation(spgt_1,133,323). coords(spgt_1,151,37,323).  visible(spgt_1,323). happensAt(walking(spgt_1),323).
orientation(spgt_0,149,324). coords(spgt_0,194,155,324).  visible(spgt_0,324). happensAt(walking(spgt_0),324). orientation(spgt_1,133,324). coords(spgt_1,152,36,324).  visible(spgt_1,324). happensAt(walking(spgt_1),324).
orientation(spgt_0,149,325). coords(spgt_0,195,156,325).  visible(spgt_0,325). happensAt(walking(spgt_0),325). orientation(spgt_1,133,325). coords(spgt_1,153,36,325).  visible(spgt_1,325). happensAt(walking(spgt_1),325).
orientation(spgt_0,149,326). coords(spgt_0,195,157,326).  visible(spgt_0,326). happensAt(walking(spgt_0),326). orientation(spgt_1,133,326). coords(spgt_1,153,36,326).  visible(spgt_1,326). happensAt(walking(spgt_1),326).
orientation(spgt_0,149,327). coords(spgt_0,195,158,327).  visible(spgt_0,327). happensAt(walking(spgt_0),327). orientation(spgt_1,133,327). coords(spgt_1,153,35,327).  visible(spgt_1,327). happensAt(walking(spgt_1),327).
orientation(spgt_0,149,328). coords(spgt_0,196,159,328).  visible(spgt_0,328). happensAt(walking(spgt_0),328). orientation(spgt_1,133,328). coords(spgt_1,154,35,328).  visible(spgt_1,328). happensAt(walking(spgt_1),328).
orientation(spgt_0,149,329). coords(spgt_0,196,160,329).  visible(spgt_0,329). happensAt(walking(spgt_0),329). orientation(spgt_1,133,329). coords(spgt_1,154,35,329).  visible(spgt_1,329). happensAt(walking(spgt_1),329).
orientation(spgt_0,149,330). coords(spgt_0,196,160,330).  visible(spgt_0,330). happensAt(walking(spgt_0),330). orientation(spgt_1,133,330). coords(spgt_1,156,35,330).  visible(spgt_1,330). happensAt(walking(spgt_1),330).
orientation(spgt_0,149,331). coords(spgt_0,200,160,331).  visible(spgt_0,331). happensAt(walking(spgt_0),331). orientation(spgt_1,133,331). coords(spgt_1,157,34,331).  visible(spgt_1,331). happensAt(walking(spgt_1),331).
orientation(spgt_0,149,332). coords(spgt_0,202,162,332).  visible(spgt_0,332). happensAt(walking(spgt_0),332). orientation(spgt_1,133,332). coords(spgt_1,157,34,332).  visible(spgt_1,332). happensAt(walking(spgt_1),332).
orientation(spgt_0,149,333). coords(spgt_0,205,166,333).  visible(spgt_0,333). happensAt(walking(spgt_0),333). orientation(spgt_1,133,333). coords(spgt_1,158,34,333).  visible(spgt_1,333). happensAt(walking(spgt_1),333).
orientation(spgt_0,149,334). coords(spgt_0,206,169,334).  visible(spgt_0,334). happensAt(walking(spgt_0),334). orientation(spgt_1,133,334). coords(spgt_1,158,34,334).  visible(spgt_1,334). happensAt(walking(spgt_1),334).
orientation(spgt_0,149,335). coords(spgt_0,207,172,335).  visible(spgt_0,335). happensAt(walking(spgt_0),335). orientation(spgt_1,133,335). coords(spgt_1,159,33,335).  visible(spgt_1,335). happensAt(walking(spgt_1),335).
orientation(spgt_0,149,336). coords(spgt_0,209,174,336).  visible(spgt_0,336). happensAt(walking(spgt_0),336). orientation(spgt_1,133,336). coords(spgt_1,159,33,336).  visible(spgt_1,336). happensAt(walking(spgt_1),336).
#show holdsAt/2.