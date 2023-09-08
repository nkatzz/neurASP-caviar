
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

	orientation(spgt_0,152,132). coords(spgt_0,95,136,132).  visible(spgt_0,132). happensAt(walking(spgt_0),132). orientation(spgt_1,154,132). coords(spgt_1,93,121,132).  visible(spgt_1,132). happensAt(walking(spgt_1),132).
orientation(spgt_0,152,133). coords(spgt_0,95,136,133).  visible(spgt_0,133). happensAt(walking(spgt_0),133). orientation(spgt_1,154,133). coords(spgt_1,94,121,133).  visible(spgt_1,133). happensAt(walking(spgt_1),133).
orientation(spgt_0,148,134). coords(spgt_0,95,135,134).  visible(spgt_0,134). happensAt(walking(spgt_0),134). orientation(spgt_1,154,134). coords(spgt_1,94,120,134).  visible(spgt_1,134). happensAt(walking(spgt_1),134).
orientation(spgt_0,148,135). coords(spgt_0,96,135,135).  visible(spgt_0,135). happensAt(walking(spgt_0),135). orientation(spgt_1,154,135). coords(spgt_1,94,120,135).  visible(spgt_1,135). happensAt(walking(spgt_1),135).
orientation(spgt_0,148,136). coords(spgt_0,96,135,136).  visible(spgt_0,136). happensAt(walking(spgt_0),136). orientation(spgt_1,154,136). coords(spgt_1,94,119,136).  visible(spgt_1,136). happensAt(walking(spgt_1),136).
orientation(spgt_0,152,137). coords(spgt_0,96,133,137).  visible(spgt_0,137). happensAt(walking(spgt_0),137). orientation(spgt_1,154,137). coords(spgt_1,95,118,137).  visible(spgt_1,137). happensAt(walking(spgt_1),137).
orientation(spgt_0,152,138). coords(spgt_0,96,132,138).  visible(spgt_0,138). happensAt(walking(spgt_0),138). orientation(spgt_1,154,138). coords(spgt_1,96,117,138).  visible(spgt_1,138). happensAt(walking(spgt_1),138).
orientation(spgt_0,153,139). coords(spgt_0,96,131,139).  visible(spgt_0,139). happensAt(walking(spgt_0),139). orientation(spgt_1,154,139). coords(spgt_1,96,117,139).  visible(spgt_1,139). happensAt(walking(spgt_1),139).
orientation(spgt_0,151,140). coords(spgt_0,97,130,140).  visible(spgt_0,140). happensAt(walking(spgt_0),140). orientation(spgt_1,148,140). coords(spgt_1,96,116,140).  visible(spgt_1,140). happensAt(walking(spgt_1),140).
orientation(spgt_0,154,141). coords(spgt_0,99,129,141).  visible(spgt_0,141). happensAt(walking(spgt_0),141). orientation(spgt_1,148,141). coords(spgt_1,96,116,141).  visible(spgt_1,141). happensAt(walking(spgt_1),141).
orientation(spgt_0,155,142). coords(spgt_0,101,129,142).  visible(spgt_0,142). happensAt(walking(spgt_0),142). orientation(spgt_1,148,142). coords(spgt_1,97,114,142).  visible(spgt_1,142). happensAt(walking(spgt_1),142).
orientation(spgt_0,155,143). coords(spgt_0,101,128,143).  visible(spgt_0,143). happensAt(walking(spgt_0),143). orientation(spgt_1,148,143). coords(spgt_1,98,114,143).  visible(spgt_1,143). happensAt(walking(spgt_1),143).
orientation(spgt_0,155,144). coords(spgt_0,103,128,144).  visible(spgt_0,144). happensAt(walking(spgt_0),144). orientation(spgt_1,148,144). coords(spgt_1,99,113,144).  visible(spgt_1,144). happensAt(walking(spgt_1),144).
orientation(spgt_0,159,145). coords(spgt_0,104,127,145).  visible(spgt_0,145). happensAt(walking(spgt_0),145). orientation(spgt_1,148,145). coords(spgt_1,99,113,145).  visible(spgt_1,145). happensAt(walking(spgt_1),145).
orientation(spgt_0,156,146). coords(spgt_0,104,127,146).  visible(spgt_0,146). happensAt(walking(spgt_0),146). orientation(spgt_1,148,146). coords(spgt_1,100,113,146).  visible(spgt_1,146). happensAt(walking(spgt_1),146).
orientation(spgt_0,154,147). coords(spgt_0,105,126,147).  visible(spgt_0,147). happensAt(walking(spgt_0),147). orientation(spgt_1,148,147). coords(spgt_1,101,113,147).  visible(spgt_1,147). happensAt(walking(spgt_1),147).
orientation(spgt_0,154,148). coords(spgt_0,105,126,148).  visible(spgt_0,148). happensAt(walking(spgt_0),148). orientation(spgt_1,148,148). coords(spgt_1,101,112,148).  visible(spgt_1,148). happensAt(walking(spgt_1),148).
orientation(spgt_0,154,149). coords(spgt_0,105,126,149).  visible(spgt_0,149). happensAt(walking(spgt_0),149). orientation(spgt_1,148,149). coords(spgt_1,102,112,149).  visible(spgt_1,149). happensAt(walking(spgt_1),149).
orientation(spgt_0,154,150). coords(spgt_0,106,125,150).  visible(spgt_0,150). happensAt(walking(spgt_0),150). orientation(spgt_1,148,150). coords(spgt_1,102,112,150).  visible(spgt_1,150). happensAt(walking(spgt_1),150).
orientation(spgt_0,154,151). coords(spgt_0,106,125,151).  visible(spgt_0,151). happensAt(walking(spgt_0),151). orientation(spgt_1,148,151). coords(spgt_1,102,111,151).  visible(spgt_1,151). happensAt(walking(spgt_1),151).
orientation(spgt_0,154,152). coords(spgt_0,106,124,152).  visible(spgt_0,152). happensAt(walking(spgt_0),152). orientation(spgt_1,148,152). coords(spgt_1,102,111,152).  visible(spgt_1,152). happensAt(walking(spgt_1),152).
orientation(spgt_0,154,153). coords(spgt_0,106,124,153).  visible(spgt_0,153). happensAt(walking(spgt_0),153). orientation(spgt_1,148,153). coords(spgt_1,103,110,153).  visible(spgt_1,153). happensAt(walking(spgt_1),153).
orientation(spgt_0,154,154). coords(spgt_0,106,123,154).  visible(spgt_0,154). happensAt(walking(spgt_0),154). orientation(spgt_1,148,154). coords(spgt_1,103,110,154).  visible(spgt_1,154). happensAt(walking(spgt_1),154).
orientation(spgt_0,154,155). coords(spgt_0,107,123,155).  visible(spgt_0,155). happensAt(walking(spgt_0),155). orientation(spgt_1,148,155). coords(spgt_1,103,110,155).  visible(spgt_1,155). happensAt(walking(spgt_1),155).
orientation(spgt_0,154,156). coords(spgt_0,107,123,156).  visible(spgt_0,156). happensAt(walking(spgt_0),156). orientation(spgt_1,148,156). coords(spgt_1,103,109,156).  visible(spgt_1,156). happensAt(walking(spgt_1),156).
#show holdsAt/2.