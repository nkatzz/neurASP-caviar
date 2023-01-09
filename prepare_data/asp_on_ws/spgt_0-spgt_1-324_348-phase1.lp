
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
orientation(spgt_0,151,337). coords(spgt_0,210,177,337).  visible(spgt_0,337). happensAt(walking(spgt_0),337). orientation(spgt_1,133,337). coords(spgt_1,161,33,337).  visible(spgt_1,337). happensAt(walking(spgt_1),337).
orientation(spgt_0,151,338). coords(spgt_0,211,177,338).  visible(spgt_0,338). happensAt(walking(spgt_0),338). orientation(spgt_1,133,338). coords(spgt_1,161,32,338).  visible(spgt_1,338). happensAt(walking(spgt_1),338).
orientation(spgt_0,155,339). coords(spgt_0,212,179,339).  visible(spgt_0,339). happensAt(walking(spgt_0),339). orientation(spgt_1,133,339). coords(spgt_1,161,31,339).  visible(spgt_1,339). happensAt(walking(spgt_1),339).
orientation(spgt_0,155,340). coords(spgt_0,212,180,340).  visible(spgt_0,340). happensAt(walking(spgt_0),340). orientation(spgt_1,133,340). coords(spgt_1,162,32,340).  visible(spgt_1,340). happensAt(walking(spgt_1),340).
orientation(spgt_0,158,341). coords(spgt_0,213,181,341).  visible(spgt_0,341). happensAt(walking(spgt_0),341). orientation(spgt_1,129,341). coords(spgt_1,162,32,341).  visible(spgt_1,341). happensAt(walking(spgt_1),341).
orientation(spgt_0,158,342). coords(spgt_0,213,182,342).  visible(spgt_0,342). happensAt(walking(spgt_0),342). orientation(spgt_1,129,342). coords(spgt_1,162,31,342).  visible(spgt_1,342). happensAt(walking(spgt_1),342).
orientation(spgt_0,156,343). coords(spgt_0,213,184,343).  visible(spgt_0,343). happensAt(walking(spgt_0),343). orientation(spgt_1,130,343). coords(spgt_1,162,30,343).  visible(spgt_1,343). happensAt(walking(spgt_1),343).
orientation(spgt_0,156,344). coords(spgt_0,214,186,344).  visible(spgt_0,344). happensAt(walking(spgt_0),344). orientation(spgt_1,127,344). coords(spgt_1,162,30,344).  visible(spgt_1,344). happensAt(walking(spgt_1),344).
orientation(spgt_0,153,345). coords(spgt_0,214,190,345).  visible(spgt_0,345). happensAt(walking(spgt_0),345). orientation(spgt_1,127,345). coords(spgt_1,164,30,345).  visible(spgt_1,345). happensAt(walking(spgt_1),345).
orientation(spgt_0,153,346). coords(spgt_0,215,195,346).  visible(spgt_0,346). happensAt(walking(spgt_0),346). orientation(spgt_1,127,346). coords(spgt_1,165,30,346).  visible(spgt_1,346). happensAt(walking(spgt_1),346).
orientation(spgt_0,153,347). coords(spgt_0,215,199,347).  visible(spgt_0,347). happensAt(walking(spgt_0),347). orientation(spgt_1,127,347). coords(spgt_1,166,30,347).  visible(spgt_1,347). happensAt(walking(spgt_1),347).
orientation(spgt_0,153,348). coords(spgt_0,216,203,348).  visible(spgt_0,348). happensAt(walking(spgt_0),348). orientation(spgt_1,127,348). coords(spgt_1,167,30,348).  visible(spgt_1,348). happensAt(walking(spgt_1),348).
#show holdsAt/2.