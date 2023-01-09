
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

	orientation(fomdgt3_1,150,330). coords(fomdgt3_1,54,100,330).  visible(fomdgt3_1,330). happensAt(inactive(fomdgt3_1),330). orientation(fomdgt3_6,130,330). coords(fomdgt3_6,212,111,330).  visible(fomdgt3_6,330). happensAt(walking(fomdgt3_6),330).
orientation(fomdgt3_1,150,331). coords(fomdgt3_1,54,100,331).  visible(fomdgt3_1,331). happensAt(inactive(fomdgt3_1),331). orientation(fomdgt3_6,130,331). coords(fomdgt3_6,212,111,331).  visible(fomdgt3_6,331). happensAt(walking(fomdgt3_6),331).
orientation(fomdgt3_1,150,332). coords(fomdgt3_1,54,100,332).  visible(fomdgt3_1,332). happensAt(inactive(fomdgt3_1),332). orientation(fomdgt3_6,130,332). coords(fomdgt3_6,212,111,332).  visible(fomdgt3_6,332). happensAt(walking(fomdgt3_6),332).
orientation(fomdgt3_1,150,333). coords(fomdgt3_1,54,100,333).  visible(fomdgt3_1,333). happensAt(inactive(fomdgt3_1),333). orientation(fomdgt3_6,130,333). coords(fomdgt3_6,212,111,333).  visible(fomdgt3_6,333). happensAt(walking(fomdgt3_6),333).
orientation(fomdgt3_1,150,334). coords(fomdgt3_1,54,100,334).  visible(fomdgt3_1,334). happensAt(inactive(fomdgt3_1),334). orientation(fomdgt3_6,130,334). coords(fomdgt3_6,212,111,334).  visible(fomdgt3_6,334). happensAt(walking(fomdgt3_6),334).
orientation(fomdgt3_1,150,335). coords(fomdgt3_1,54,100,335).  visible(fomdgt3_1,335). happensAt(inactive(fomdgt3_1),335). orientation(fomdgt3_6,130,335). coords(fomdgt3_6,212,111,335).  visible(fomdgt3_6,335). happensAt(walking(fomdgt3_6),335).
orientation(fomdgt3_1,150,336). coords(fomdgt3_1,54,100,336).  visible(fomdgt3_1,336). happensAt(inactive(fomdgt3_1),336). orientation(fomdgt3_6,130,336). coords(fomdgt3_6,212,111,336).  visible(fomdgt3_6,336). happensAt(walking(fomdgt3_6),336).
orientation(fomdgt3_1,150,337). coords(fomdgt3_1,54,100,337).  visible(fomdgt3_1,337). happensAt(inactive(fomdgt3_1),337). orientation(fomdgt3_6,130,337). coords(fomdgt3_6,212,111,337).  visible(fomdgt3_6,337). happensAt(walking(fomdgt3_6),337).
orientation(fomdgt3_1,150,338). coords(fomdgt3_1,54,100,338).  visible(fomdgt3_1,338). happensAt(inactive(fomdgt3_1),338). orientation(fomdgt3_6,130,338). coords(fomdgt3_6,212,110,338).  visible(fomdgt3_6,338). happensAt(walking(fomdgt3_6),338).
orientation(fomdgt3_1,150,339). coords(fomdgt3_1,54,100,339).  visible(fomdgt3_1,339). happensAt(inactive(fomdgt3_1),339). orientation(fomdgt3_6,130,339). coords(fomdgt3_6,212,110,339).  visible(fomdgt3_6,339). happensAt(walking(fomdgt3_6),339).
orientation(fomdgt3_1,150,340). coords(fomdgt3_1,54,100,340).  visible(fomdgt3_1,340). happensAt(inactive(fomdgt3_1),340). orientation(fomdgt3_6,130,340). coords(fomdgt3_6,214,110,340).  visible(fomdgt3_6,340). happensAt(walking(fomdgt3_6),340).
orientation(fomdgt3_1,150,341). coords(fomdgt3_1,54,100,341).  visible(fomdgt3_1,341). happensAt(inactive(fomdgt3_1),341). orientation(fomdgt3_6,130,341). coords(fomdgt3_6,214,110,341).  visible(fomdgt3_6,341). happensAt(walking(fomdgt3_6),341).
orientation(fomdgt3_1,150,342). coords(fomdgt3_1,54,100,342).  visible(fomdgt3_1,342). happensAt(inactive(fomdgt3_1),342). orientation(fomdgt3_6,130,342). coords(fomdgt3_6,217,110,342).  visible(fomdgt3_6,342). happensAt(walking(fomdgt3_6),342).
orientation(fomdgt3_1,150,343). coords(fomdgt3_1,54,100,343).  visible(fomdgt3_1,343). happensAt(inactive(fomdgt3_1),343). orientation(fomdgt3_6,130,343). coords(fomdgt3_6,217,109,343).  visible(fomdgt3_6,343). happensAt(walking(fomdgt3_6),343).
orientation(fomdgt3_1,150,344). coords(fomdgt3_1,54,100,344).  visible(fomdgt3_1,344). happensAt(inactive(fomdgt3_1),344). orientation(fomdgt3_6,130,344). coords(fomdgt3_6,217,109,344).  visible(fomdgt3_6,344). happensAt(walking(fomdgt3_6),344).
orientation(fomdgt3_1,150,345). coords(fomdgt3_1,54,100,345).  visible(fomdgt3_1,345). happensAt(inactive(fomdgt3_1),345). orientation(fomdgt3_6,130,345). coords(fomdgt3_6,217,107,345).  visible(fomdgt3_6,345). happensAt(walking(fomdgt3_6),345).
orientation(fomdgt3_1,150,346). coords(fomdgt3_1,54,100,346).  visible(fomdgt3_1,346). happensAt(inactive(fomdgt3_1),346). orientation(fomdgt3_6,130,346). coords(fomdgt3_6,214,106,346).  visible(fomdgt3_6,346). happensAt(walking(fomdgt3_6),346).
orientation(fomdgt3_1,150,347). coords(fomdgt3_1,54,100,347).  visible(fomdgt3_1,347). happensAt(inactive(fomdgt3_1),347). orientation(fomdgt3_6,130,347). coords(fomdgt3_6,212,106,347).  visible(fomdgt3_6,347). happensAt(walking(fomdgt3_6),347).
orientation(fomdgt3_1,150,348). coords(fomdgt3_1,54,100,348).  visible(fomdgt3_1,348). happensAt(inactive(fomdgt3_1),348). orientation(fomdgt3_6,130,348). coords(fomdgt3_6,207,106,348).  visible(fomdgt3_6,348). happensAt(walking(fomdgt3_6),348).
orientation(fomdgt3_1,150,349). coords(fomdgt3_1,54,100,349).  visible(fomdgt3_1,349). happensAt(inactive(fomdgt3_1),349). orientation(fomdgt3_6,150,349). coords(fomdgt3_6,205,106,349).  visible(fomdgt3_6,349). happensAt(walking(fomdgt3_6),349).
orientation(fomdgt3_1,150,350). coords(fomdgt3_1,54,100,350).  visible(fomdgt3_1,350). happensAt(inactive(fomdgt3_1),350). orientation(fomdgt3_6,150,350). coords(fomdgt3_6,200,103,350).  visible(fomdgt3_6,350). happensAt(walking(fomdgt3_6),350).
orientation(fomdgt3_1,150,351). coords(fomdgt3_1,54,100,351).  visible(fomdgt3_1,351). happensAt(inactive(fomdgt3_1),351). orientation(fomdgt3_6,150,351). coords(fomdgt3_6,199,102,351).  visible(fomdgt3_6,351). happensAt(walking(fomdgt3_6),351).
orientation(fomdgt3_1,150,352). coords(fomdgt3_1,54,100,352).  visible(fomdgt3_1,352). happensAt(inactive(fomdgt3_1),352). orientation(fomdgt3_6,150,352). coords(fomdgt3_6,196,100,352).  visible(fomdgt3_6,352). happensAt(walking(fomdgt3_6),352).
orientation(fomdgt3_1,150,353). coords(fomdgt3_1,54,100,353).  visible(fomdgt3_1,353). happensAt(inactive(fomdgt3_1),353). orientation(fomdgt3_6,150,353). coords(fomdgt3_6,193,100,353).  visible(fomdgt3_6,353). happensAt(walking(fomdgt3_6),353).
orientation(fomdgt3_1,150,354). coords(fomdgt3_1,54,100,354).  visible(fomdgt3_1,354). happensAt(inactive(fomdgt3_1),354). orientation(fomdgt3_6,150,354). coords(fomdgt3_6,193,100,354).  visible(fomdgt3_6,354). happensAt(walking(fomdgt3_6),354).
#show holdsAt/2.