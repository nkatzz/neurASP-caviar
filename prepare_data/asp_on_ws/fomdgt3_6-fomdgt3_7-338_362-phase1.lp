
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

	orientation(fomdgt3_6,130,338). coords(fomdgt3_6,212,110,338).  visible(fomdgt3_6,338). happensAt(walking(fomdgt3_6),338). orientation(fomdgt3_7,134,338). coords(fomdgt3_7,83,67,338).  happensAt(appear(fomdgt3_7),338). happensAt(walking(fomdgt3_7),338).
orientation(fomdgt3_6,130,339). coords(fomdgt3_6,212,110,339).  visible(fomdgt3_6,339). happensAt(walking(fomdgt3_6),339). orientation(fomdgt3_7,134,339). coords(fomdgt3_7,83,67,339).  visible(fomdgt3_7,339). happensAt(walking(fomdgt3_7),339).
orientation(fomdgt3_6,130,340). coords(fomdgt3_6,214,110,340).  visible(fomdgt3_6,340). happensAt(walking(fomdgt3_6),340). orientation(fomdgt3_7,134,340). coords(fomdgt3_7,83,67,340).  visible(fomdgt3_7,340). happensAt(walking(fomdgt3_7),340).
orientation(fomdgt3_6,130,341). coords(fomdgt3_6,214,110,341).  visible(fomdgt3_6,341). happensAt(walking(fomdgt3_6),341). orientation(fomdgt3_7,134,341). coords(fomdgt3_7,83,67,341).  visible(fomdgt3_7,341). happensAt(walking(fomdgt3_7),341).
orientation(fomdgt3_6,130,342). coords(fomdgt3_6,217,110,342).  visible(fomdgt3_6,342). happensAt(walking(fomdgt3_6),342). orientation(fomdgt3_7,134,342). coords(fomdgt3_7,82,67,342).  visible(fomdgt3_7,342). happensAt(walking(fomdgt3_7),342).
orientation(fomdgt3_6,130,343). coords(fomdgt3_6,217,109,343).  visible(fomdgt3_6,343). happensAt(walking(fomdgt3_6),343). orientation(fomdgt3_7,134,343). coords(fomdgt3_7,82,67,343).  visible(fomdgt3_7,343). happensAt(walking(fomdgt3_7),343).
orientation(fomdgt3_6,130,344). coords(fomdgt3_6,217,109,344).  visible(fomdgt3_6,344). happensAt(walking(fomdgt3_6),344). orientation(fomdgt3_7,134,344). coords(fomdgt3_7,82,67,344).  visible(fomdgt3_7,344). happensAt(walking(fomdgt3_7),344).
orientation(fomdgt3_6,130,345). coords(fomdgt3_6,217,107,345).  visible(fomdgt3_6,345). happensAt(walking(fomdgt3_6),345). orientation(fomdgt3_7,134,345). coords(fomdgt3_7,82,67,345).  visible(fomdgt3_7,345). happensAt(walking(fomdgt3_7),345).
orientation(fomdgt3_6,130,346). coords(fomdgt3_6,214,106,346).  visible(fomdgt3_6,346). happensAt(walking(fomdgt3_6),346). orientation(fomdgt3_7,134,346). coords(fomdgt3_7,81,69,346).  visible(fomdgt3_7,346). happensAt(walking(fomdgt3_7),346).
orientation(fomdgt3_6,130,347). coords(fomdgt3_6,212,106,347).  visible(fomdgt3_6,347). happensAt(walking(fomdgt3_6),347). orientation(fomdgt3_7,134,347). coords(fomdgt3_7,81,69,347).  visible(fomdgt3_7,347). happensAt(walking(fomdgt3_7),347).
orientation(fomdgt3_6,130,348). coords(fomdgt3_6,207,106,348).  visible(fomdgt3_6,348). happensAt(walking(fomdgt3_6),348). orientation(fomdgt3_7,134,348). coords(fomdgt3_7,79,69,348).  visible(fomdgt3_7,348). happensAt(walking(fomdgt3_7),348).
orientation(fomdgt3_6,150,349). coords(fomdgt3_6,205,106,349).  visible(fomdgt3_6,349). happensAt(walking(fomdgt3_6),349). orientation(fomdgt3_7,134,349). coords(fomdgt3_7,78,70,349).  visible(fomdgt3_7,349). happensAt(walking(fomdgt3_7),349).
orientation(fomdgt3_6,150,350). coords(fomdgt3_6,200,103,350).  visible(fomdgt3_6,350). happensAt(walking(fomdgt3_6),350). orientation(fomdgt3_7,134,350). coords(fomdgt3_7,78,71,350).  visible(fomdgt3_7,350). happensAt(walking(fomdgt3_7),350).
orientation(fomdgt3_6,150,351). coords(fomdgt3_6,199,102,351).  visible(fomdgt3_6,351). happensAt(walking(fomdgt3_6),351). orientation(fomdgt3_7,134,351). coords(fomdgt3_7,78,71,351).  visible(fomdgt3_7,351). happensAt(walking(fomdgt3_7),351).
orientation(fomdgt3_6,150,352). coords(fomdgt3_6,196,100,352).  visible(fomdgt3_6,352). happensAt(walking(fomdgt3_6),352). orientation(fomdgt3_7,134,352). coords(fomdgt3_7,78,71,352).  visible(fomdgt3_7,352). happensAt(walking(fomdgt3_7),352).
orientation(fomdgt3_6,150,353). coords(fomdgt3_6,193,100,353).  visible(fomdgt3_6,353). happensAt(walking(fomdgt3_6),353). orientation(fomdgt3_7,134,353). coords(fomdgt3_7,78,75,353).  visible(fomdgt3_7,353). happensAt(walking(fomdgt3_7),353).
orientation(fomdgt3_6,150,354). coords(fomdgt3_6,193,100,354).  visible(fomdgt3_6,354). happensAt(walking(fomdgt3_6),354). orientation(fomdgt3_7,134,354). coords(fomdgt3_7,78,75,354).  visible(fomdgt3_7,354). happensAt(walking(fomdgt3_7),354).
orientation(fomdgt3_6,150,355). coords(fomdgt3_6,193,93,355).  visible(fomdgt3_6,355). happensAt(walking(fomdgt3_6),355). orientation(fomdgt3_7,134,355). coords(fomdgt3_7,78,75,355).  visible(fomdgt3_7,355). happensAt(walking(fomdgt3_7),355).
orientation(fomdgt3_6,150,356). coords(fomdgt3_6,186,90,356).  visible(fomdgt3_6,356). happensAt(walking(fomdgt3_6),356). orientation(fomdgt3_7,134,356). coords(fomdgt3_7,78,76,356).  visible(fomdgt3_7,356). happensAt(walking(fomdgt3_7),356).
orientation(fomdgt3_6,156,357). coords(fomdgt3_6,184,90,357).  visible(fomdgt3_6,357). happensAt(walking(fomdgt3_6),357). orientation(fomdgt3_7,134,357). coords(fomdgt3_7,78,76,357).  visible(fomdgt3_7,357). happensAt(walking(fomdgt3_7),357).
orientation(fomdgt3_6,156,358). coords(fomdgt3_6,182,90,358).  visible(fomdgt3_6,358). happensAt(walking(fomdgt3_6),358). orientation(fomdgt3_7,134,358). coords(fomdgt3_7,78,76,358).  visible(fomdgt3_7,358). happensAt(walking(fomdgt3_7),358).
orientation(fomdgt3_6,156,359). coords(fomdgt3_6,181,90,359).  visible(fomdgt3_6,359). happensAt(walking(fomdgt3_6),359). orientation(fomdgt3_7,134,359). coords(fomdgt3_7,78,76,359).  visible(fomdgt3_7,359). happensAt(walking(fomdgt3_7),359).
orientation(fomdgt3_6,156,360). coords(fomdgt3_6,178,94,360).  visible(fomdgt3_6,360). happensAt(walking(fomdgt3_6),360). orientation(fomdgt3_7,134,360). coords(fomdgt3_7,78,76,360).  visible(fomdgt3_7,360). happensAt(walking(fomdgt3_7),360).
orientation(fomdgt3_6,156,361). coords(fomdgt3_6,176,94,361).  visible(fomdgt3_6,361). happensAt(walking(fomdgt3_6),361). orientation(fomdgt3_7,134,361). coords(fomdgt3_7,78,76,361).  visible(fomdgt3_7,361). happensAt(walking(fomdgt3_7),361).
orientation(fomdgt3_6,156,362). coords(fomdgt3_6,170,95,362).  visible(fomdgt3_6,362). happensAt(walking(fomdgt3_6),362). orientation(fomdgt3_7,134,362). coords(fomdgt3_7,78,76,362).  visible(fomdgt3_7,362). happensAt(walking(fomdgt3_7),362).
#show holdsAt/2.