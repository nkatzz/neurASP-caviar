
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

	orientation(rsfgt_0,160,132). coords(rsfgt_0,41,155,132).  visible(rsfgt_0,132). happensAt(walking(rsfgt_0),132). orientation(rsfgt_1,150,132). coords(rsfgt_1,166,108,132).  visible(rsfgt_1,132). happensAt(active(rsfgt_1),132).
orientation(rsfgt_0,160,133). coords(rsfgt_0,39,155,133).  visible(rsfgt_0,133). happensAt(walking(rsfgt_0),133). orientation(rsfgt_1,150,133). coords(rsfgt_1,166,108,133).  visible(rsfgt_1,133). happensAt(active(rsfgt_1),133).
orientation(rsfgt_0,160,134). coords(rsfgt_0,39,155,134).  visible(rsfgt_0,134). happensAt(walking(rsfgt_0),134). orientation(rsfgt_1,150,134). coords(rsfgt_1,166,108,134).  visible(rsfgt_1,134). happensAt(active(rsfgt_1),134).
orientation(rsfgt_0,160,135). coords(rsfgt_0,39,155,135).  visible(rsfgt_0,135). happensAt(walking(rsfgt_0),135). orientation(rsfgt_1,150,135). coords(rsfgt_1,166,108,135).  visible(rsfgt_1,135). happensAt(active(rsfgt_1),135).
orientation(rsfgt_0,160,136). coords(rsfgt_0,39,155,136).  visible(rsfgt_0,136). happensAt(walking(rsfgt_0),136). orientation(rsfgt_1,150,136). coords(rsfgt_1,166,108,136).  visible(rsfgt_1,136). happensAt(active(rsfgt_1),136).
orientation(rsfgt_0,160,137). coords(rsfgt_0,39,155,137).  visible(rsfgt_0,137). happensAt(walking(rsfgt_0),137). orientation(rsfgt_1,150,137). coords(rsfgt_1,166,108,137).  visible(rsfgt_1,137). happensAt(active(rsfgt_1),137).
orientation(rsfgt_0,160,138). coords(rsfgt_0,38,155,138).  visible(rsfgt_0,138). happensAt(walking(rsfgt_0),138). orientation(rsfgt_1,150,138). coords(rsfgt_1,166,108,138).  visible(rsfgt_1,138). happensAt(active(rsfgt_1),138).
orientation(rsfgt_0,160,139). coords(rsfgt_0,38,155,139).  visible(rsfgt_0,139). happensAt(walking(rsfgt_0),139). orientation(rsfgt_1,150,139). coords(rsfgt_1,166,108,139).  visible(rsfgt_1,139). happensAt(active(rsfgt_1),139).
orientation(rsfgt_0,160,140). coords(rsfgt_0,38,155,140).  visible(rsfgt_0,140). happensAt(walking(rsfgt_0),140). orientation(rsfgt_1,150,140). coords(rsfgt_1,166,108,140).  visible(rsfgt_1,140). happensAt(active(rsfgt_1),140).
orientation(rsfgt_0,160,141). coords(rsfgt_0,38,155,141).  visible(rsfgt_0,141). happensAt(walking(rsfgt_0),141). orientation(rsfgt_1,150,141). coords(rsfgt_1,166,108,141).  visible(rsfgt_1,141). happensAt(active(rsfgt_1),141).
orientation(rsfgt_0,160,142). coords(rsfgt_0,38,155,142).  visible(rsfgt_0,142). happensAt(walking(rsfgt_0),142). orientation(rsfgt_1,150,142). coords(rsfgt_1,168,111,142).  visible(rsfgt_1,142). happensAt(walking(rsfgt_1),142).
orientation(rsfgt_0,160,143). coords(rsfgt_0,38,155,143).  visible(rsfgt_0,143). happensAt(walking(rsfgt_0),143). orientation(rsfgt_1,150,143). coords(rsfgt_1,168,113,143).  visible(rsfgt_1,143). happensAt(walking(rsfgt_1),143).
orientation(rsfgt_0,160,144). coords(rsfgt_0,35,155,144).  visible(rsfgt_0,144). happensAt(walking(rsfgt_0),144). orientation(rsfgt_1,150,144). coords(rsfgt_1,171,113,144).  visible(rsfgt_1,144). happensAt(walking(rsfgt_1),144).
orientation(rsfgt_0,160,145). coords(rsfgt_0,35,155,145).  visible(rsfgt_0,145). happensAt(walking(rsfgt_0),145). orientation(rsfgt_1,150,145). coords(rsfgt_1,171,116,145).  visible(rsfgt_1,145). happensAt(walking(rsfgt_1),145).
orientation(rsfgt_0,160,146). coords(rsfgt_0,35,155,146).  visible(rsfgt_0,146). happensAt(walking(rsfgt_0),146). orientation(rsfgt_1,150,146). coords(rsfgt_1,175,117,146).  visible(rsfgt_1,146). happensAt(walking(rsfgt_1),146).
orientation(rsfgt_0,160,147). coords(rsfgt_0,35,155,147).  visible(rsfgt_0,147). happensAt(walking(rsfgt_0),147). orientation(rsfgt_1,150,147). coords(rsfgt_1,176,119,147).  visible(rsfgt_1,147). happensAt(walking(rsfgt_1),147).
orientation(rsfgt_0,160,148). coords(rsfgt_0,35,155,148).  visible(rsfgt_0,148). happensAt(walking(rsfgt_0),148). orientation(rsfgt_1,150,148). coords(rsfgt_1,176,119,148).  visible(rsfgt_1,148). happensAt(walking(rsfgt_1),148).
orientation(rsfgt_0,160,149). coords(rsfgt_0,35,155,149).  visible(rsfgt_0,149). happensAt(walking(rsfgt_0),149). orientation(rsfgt_1,150,149). coords(rsfgt_1,176,119,149).  visible(rsfgt_1,149). happensAt(walking(rsfgt_1),149).
orientation(rsfgt_0,160,150). coords(rsfgt_0,35,155,150).  visible(rsfgt_0,150). happensAt(walking(rsfgt_0),150). orientation(rsfgt_1,150,150). coords(rsfgt_1,176,124,150).  visible(rsfgt_1,150). happensAt(walking(rsfgt_1),150).
orientation(rsfgt_0,160,151). coords(rsfgt_0,35,155,151).  visible(rsfgt_0,151). happensAt(walking(rsfgt_0),151). orientation(rsfgt_1,150,151). coords(rsfgt_1,176,128,151).  visible(rsfgt_1,151). happensAt(walking(rsfgt_1),151).
orientation(rsfgt_0,160,152). coords(rsfgt_0,35,155,152).  visible(rsfgt_0,152). happensAt(walking(rsfgt_0),152). orientation(rsfgt_1,150,152). coords(rsfgt_1,176,129,152).  visible(rsfgt_1,152). happensAt(walking(rsfgt_1),152).
orientation(rsfgt_0,160,153). coords(rsfgt_0,35,155,153).  visible(rsfgt_0,153). happensAt(walking(rsfgt_0),153). orientation(rsfgt_1,150,153). coords(rsfgt_1,176,129,153).  visible(rsfgt_1,153). happensAt(walking(rsfgt_1),153).
orientation(rsfgt_0,160,154). coords(rsfgt_0,34,155,154).  visible(rsfgt_0,154). happensAt(walking(rsfgt_0),154). orientation(rsfgt_1,150,154). coords(rsfgt_1,179,130,154).  visible(rsfgt_1,154). happensAt(walking(rsfgt_1),154).
orientation(rsfgt_0,160,155). coords(rsfgt_0,34,155,155).  visible(rsfgt_0,155). happensAt(walking(rsfgt_0),155). orientation(rsfgt_1,150,155). coords(rsfgt_1,179,131,155).  visible(rsfgt_1,155). happensAt(walking(rsfgt_1),155).
orientation(rsfgt_0,160,156). coords(rsfgt_0,34,155,156).  visible(rsfgt_0,156). happensAt(walking(rsfgt_0),156). orientation(rsfgt_1,150,156). coords(rsfgt_1,179,131,156).  visible(rsfgt_1,156). happensAt(walking(rsfgt_1),156).
#show holdsAt/2.