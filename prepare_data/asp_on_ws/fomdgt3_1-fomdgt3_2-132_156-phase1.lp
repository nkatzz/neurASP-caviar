
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

	orientation(fomdgt3_1,150,132). coords(fomdgt3_1,121,108,132).  visible(fomdgt3_1,132). happensAt(walking(fomdgt3_1),132). orientation(fomdgt3_2,50,132). coords(fomdgt3_2,275,262,132).  visible(fomdgt3_2,132). happensAt(walking(fomdgt3_2),132).
orientation(fomdgt3_1,150,133). coords(fomdgt3_1,121,108,133).  visible(fomdgt3_1,133). happensAt(walking(fomdgt3_1),133). orientation(fomdgt3_2,50,133). coords(fomdgt3_2,275,262,133).  visible(fomdgt3_2,133). happensAt(walking(fomdgt3_2),133).
orientation(fomdgt3_1,150,134). coords(fomdgt3_1,120,108,134).  visible(fomdgt3_1,134). happensAt(walking(fomdgt3_1),134). orientation(fomdgt3_2,40,134). coords(fomdgt3_2,274,258,134).  visible(fomdgt3_2,134). happensAt(walking(fomdgt3_2),134).
orientation(fomdgt3_1,150,135). coords(fomdgt3_1,120,108,135).  visible(fomdgt3_1,135). happensAt(walking(fomdgt3_1),135). orientation(fomdgt3_2,40,135). coords(fomdgt3_2,274,256,135).  visible(fomdgt3_2,135). happensAt(walking(fomdgt3_2),135).
orientation(fomdgt3_1,150,136). coords(fomdgt3_1,120,108,136).  visible(fomdgt3_1,136). happensAt(walking(fomdgt3_1),136). orientation(fomdgt3_2,40,136). coords(fomdgt3_2,274,253,136).  visible(fomdgt3_2,136). happensAt(walking(fomdgt3_2),136).
orientation(fomdgt3_1,150,137). coords(fomdgt3_1,120,108,137).  visible(fomdgt3_1,137). happensAt(walking(fomdgt3_1),137). orientation(fomdgt3_2,40,137). coords(fomdgt3_2,273,249,137).  visible(fomdgt3_2,137). happensAt(walking(fomdgt3_2),137).
orientation(fomdgt3_1,150,138). coords(fomdgt3_1,118,108,138).  visible(fomdgt3_1,138). happensAt(walking(fomdgt3_1),138). orientation(fomdgt3_2,40,138). coords(fomdgt3_2,273,247,138).  visible(fomdgt3_2,138). happensAt(walking(fomdgt3_2),138).
orientation(fomdgt3_1,150,139). coords(fomdgt3_1,118,108,139).  visible(fomdgt3_1,139). happensAt(walking(fomdgt3_1),139). orientation(fomdgt3_2,40,139). coords(fomdgt3_2,272,246,139).  visible(fomdgt3_2,139). happensAt(walking(fomdgt3_2),139).
orientation(fomdgt3_1,150,140). coords(fomdgt3_1,118,108,140).  visible(fomdgt3_1,140). happensAt(walking(fomdgt3_1),140). orientation(fomdgt3_2,40,140). coords(fomdgt3_2,271,245,140).  visible(fomdgt3_2,140). happensAt(walking(fomdgt3_2),140).
orientation(fomdgt3_1,150,141). coords(fomdgt3_1,116,108,141).  visible(fomdgt3_1,141). happensAt(walking(fomdgt3_1),141). orientation(fomdgt3_2,40,141). coords(fomdgt3_2,271,244,141).  visible(fomdgt3_2,141). happensAt(walking(fomdgt3_2),141).
orientation(fomdgt3_1,150,142). coords(fomdgt3_1,116,108,142).  visible(fomdgt3_1,142). happensAt(walking(fomdgt3_1),142). orientation(fomdgt3_2,40,142). coords(fomdgt3_2,271,244,142).  visible(fomdgt3_2,142). happensAt(walking(fomdgt3_2),142).
orientation(fomdgt3_1,150,143). coords(fomdgt3_1,113,108,143).  visible(fomdgt3_1,143). happensAt(walking(fomdgt3_1),143). orientation(fomdgt3_2,40,143). coords(fomdgt3_2,271,244,143).  visible(fomdgt3_2,143). happensAt(walking(fomdgt3_2),143).
orientation(fomdgt3_1,150,144). coords(fomdgt3_1,113,108,144).  visible(fomdgt3_1,144). happensAt(walking(fomdgt3_1),144). orientation(fomdgt3_2,5,144). coords(fomdgt3_2,268,244,144).  visible(fomdgt3_2,144). happensAt(walking(fomdgt3_2),144).
orientation(fomdgt3_1,150,145). coords(fomdgt3_1,110,108,145).  visible(fomdgt3_1,145). happensAt(walking(fomdgt3_1),145). orientation(fomdgt3_2,5,145). coords(fomdgt3_2,268,244,145).  visible(fomdgt3_2,145). happensAt(walking(fomdgt3_2),145).
orientation(fomdgt3_1,150,146). coords(fomdgt3_1,110,108,146).  visible(fomdgt3_1,146). happensAt(walking(fomdgt3_1),146). orientation(fomdgt3_2,10,146). coords(fomdgt3_2,268,238,146).  visible(fomdgt3_2,146). happensAt(walking(fomdgt3_2),146).
orientation(fomdgt3_1,150,147). coords(fomdgt3_1,110,108,147).  visible(fomdgt3_1,147). happensAt(walking(fomdgt3_1),147). orientation(fomdgt3_2,10,147). coords(fomdgt3_2,265,235,147).  visible(fomdgt3_2,147). happensAt(walking(fomdgt3_2),147).
orientation(fomdgt3_1,150,148). coords(fomdgt3_1,110,108,148).  visible(fomdgt3_1,148). happensAt(walking(fomdgt3_1),148). orientation(fomdgt3_2,10,148). coords(fomdgt3_2,265,232,148).  visible(fomdgt3_2,148). happensAt(walking(fomdgt3_2),148).
orientation(fomdgt3_1,150,149). coords(fomdgt3_1,110,108,149).  visible(fomdgt3_1,149). happensAt(walking(fomdgt3_1),149). orientation(fomdgt3_2,10,149). coords(fomdgt3_2,262,230,149).  visible(fomdgt3_2,149). happensAt(walking(fomdgt3_2),149).
orientation(fomdgt3_1,150,150). coords(fomdgt3_1,107,108,150).  visible(fomdgt3_1,150). happensAt(walking(fomdgt3_1),150). orientation(fomdgt3_2,10,150). coords(fomdgt3_2,262,228,150).  visible(fomdgt3_2,150). happensAt(walking(fomdgt3_2),150).
orientation(fomdgt3_1,150,151). coords(fomdgt3_1,107,108,151).  visible(fomdgt3_1,151). happensAt(walking(fomdgt3_1),151). orientation(fomdgt3_2,10,151). coords(fomdgt3_2,260,226,151).  visible(fomdgt3_2,151). happensAt(walking(fomdgt3_2),151).
orientation(fomdgt3_1,150,152). coords(fomdgt3_1,106,108,152).  visible(fomdgt3_1,152). happensAt(walking(fomdgt3_1),152). orientation(fomdgt3_2,18,152). coords(fomdgt3_2,260,223,152).  visible(fomdgt3_2,152). happensAt(walking(fomdgt3_2),152).
orientation(fomdgt3_1,150,153). coords(fomdgt3_1,106,108,153).  visible(fomdgt3_1,153). happensAt(walking(fomdgt3_1),153). orientation(fomdgt3_2,18,153). coords(fomdgt3_2,260,221,153).  visible(fomdgt3_2,153). happensAt(walking(fomdgt3_2),153).
orientation(fomdgt3_1,150,154). coords(fomdgt3_1,106,108,154).  visible(fomdgt3_1,154). happensAt(walking(fomdgt3_1),154). orientation(fomdgt3_2,18,154). coords(fomdgt3_2,260,221,154).  visible(fomdgt3_2,154). happensAt(walking(fomdgt3_2),154).
orientation(fomdgt3_1,150,155). coords(fomdgt3_1,106,108,155).  visible(fomdgt3_1,155). happensAt(walking(fomdgt3_1),155). orientation(fomdgt3_2,18,155). coords(fomdgt3_2,260,221,155).  visible(fomdgt3_2,155). happensAt(walking(fomdgt3_2),155).
orientation(fomdgt3_1,150,156). coords(fomdgt3_1,106,108,156).  visible(fomdgt3_1,156). happensAt(walking(fomdgt3_1),156). orientation(fomdgt3_2,18,156). coords(fomdgt3_2,260,221,156).  visible(fomdgt3_2,156). happensAt(walking(fomdgt3_2),156).
#show holdsAt/2.