
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
orientation(spgt_0,153,349). coords(spgt_0,217,203,349).  visible(spgt_0,349). happensAt(walking(spgt_0),349). orientation(spgt_1,127,349). coords(spgt_1,168,28,349).  visible(spgt_1,349). happensAt(walking(spgt_1),349).
orientation(spgt_0,154,350). coords(spgt_0,218,204,350).  visible(spgt_0,350). happensAt(walking(spgt_0),350). orientation(spgt_1,127,350). coords(spgt_1,168,27,350).  visible(spgt_1,350). happensAt(walking(spgt_1),350).
orientation(spgt_0,159,351). coords(spgt_0,218,204,351).  visible(spgt_0,351). happensAt(walking(spgt_0),351). orientation(spgt_1,127,351). coords(spgt_1,168,27,351).  visible(spgt_1,351). happensAt(walking(spgt_1),351).
orientation(spgt_0,165,352). coords(spgt_0,219,205,352).  visible(spgt_0,352). happensAt(walking(spgt_0),352). orientation(spgt_1,127,352). coords(spgt_1,169,27,352).  visible(spgt_1,352). happensAt(walking(spgt_1),352).
orientation(spgt_0,165,353). coords(spgt_0,219,207,353).  visible(spgt_0,353). happensAt(walking(spgt_0),353). orientation(spgt_1,127,353). coords(spgt_1,169,27,353).  visible(spgt_1,353). happensAt(walking(spgt_1),353).
orientation(spgt_0,169,354). coords(spgt_0,219,209,354).  visible(spgt_0,354). happensAt(walking(spgt_0),354). orientation(spgt_1,127,354). coords(spgt_1,169,27,354).  visible(spgt_1,354). happensAt(walking(spgt_1),354).
orientation(spgt_0,172,355). coords(spgt_0,219,211,355).  visible(spgt_0,355). happensAt(walking(spgt_0),355). orientation(spgt_1,127,355). coords(spgt_1,170,27,355).  visible(spgt_1,355). happensAt(walking(spgt_1),355).
orientation(spgt_0,172,356). coords(spgt_0,221,213,356).  visible(spgt_0,356). happensAt(walking(spgt_0),356). orientation(spgt_1,127,356). coords(spgt_1,170,27,356).  visible(spgt_1,356). happensAt(walking(spgt_1),356).
orientation(spgt_0,172,357). coords(spgt_0,223,215,357).  visible(spgt_0,357). happensAt(walking(spgt_0),357). orientation(spgt_1,123,357). coords(spgt_1,170,27,357).  visible(spgt_1,357). happensAt(walking(spgt_1),357).
orientation(spgt_0,1,358). coords(spgt_0,226,217,358).  visible(spgt_0,358). happensAt(walking(spgt_0),358). orientation(spgt_1,123,358). coords(spgt_1,170,27,358).  visible(spgt_1,358). happensAt(walking(spgt_1),358).
orientation(spgt_0,1,359). coords(spgt_0,227,219,359).  visible(spgt_0,359). happensAt(walking(spgt_0),359). orientation(spgt_1,123,359). coords(spgt_1,170,27,359).  visible(spgt_1,359). happensAt(walking(spgt_1),359).
orientation(spgt_0,1,360). coords(spgt_0,230,222,360).  visible(spgt_0,360). happensAt(walking(spgt_0),360). orientation(spgt_1,123,360). coords(spgt_1,170,27,360).  visible(spgt_1,360). happensAt(walking(spgt_1),360).
#show holdsAt/2.