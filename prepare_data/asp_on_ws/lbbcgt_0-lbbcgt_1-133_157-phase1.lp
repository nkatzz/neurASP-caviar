
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

	orientation(lbbcgt_0,12,133). coords(lbbcgt_0,222,235,133).  visible(lbbcgt_0,133). happensAt(walking(lbbcgt_0),133). orientation(lbbcgt_1,147,133). coords(lbbcgt_1,13,56,133).  visible(lbbcgt_1,133). happensAt(walking(lbbcgt_1),133).
orientation(lbbcgt_0,12,134). coords(lbbcgt_0,225,238,134).  visible(lbbcgt_0,134). happensAt(walking(lbbcgt_0),134). orientation(lbbcgt_1,147,134). coords(lbbcgt_1,13,56,134).  visible(lbbcgt_1,134). happensAt(walking(lbbcgt_1),134).
orientation(lbbcgt_0,22,135). coords(lbbcgt_0,227,240,135).  visible(lbbcgt_0,135). happensAt(walking(lbbcgt_0),135). orientation(lbbcgt_1,147,135). coords(lbbcgt_1,13,56,135).  visible(lbbcgt_1,135). happensAt(walking(lbbcgt_1),135).
orientation(lbbcgt_0,22,136). coords(lbbcgt_0,229,242,136).  visible(lbbcgt_0,136). happensAt(walking(lbbcgt_0),136). orientation(lbbcgt_1,147,136). coords(lbbcgt_1,13,56,136).  visible(lbbcgt_1,136). happensAt(walking(lbbcgt_1),136).
orientation(lbbcgt_0,22,137). coords(lbbcgt_0,230,245,137).  visible(lbbcgt_0,137). happensAt(walking(lbbcgt_0),137). orientation(lbbcgt_1,147,137). coords(lbbcgt_1,13,57,137).  visible(lbbcgt_1,137). happensAt(walking(lbbcgt_1),137).
orientation(lbbcgt_0,22,138). coords(lbbcgt_0,232,247,138).  visible(lbbcgt_0,138). happensAt(walking(lbbcgt_0),138). orientation(lbbcgt_1,147,138). coords(lbbcgt_1,13,58,138).  visible(lbbcgt_1,138). happensAt(walking(lbbcgt_1),138).
orientation(lbbcgt_0,22,139). coords(lbbcgt_0,233,249,139).  visible(lbbcgt_0,139). happensAt(walking(lbbcgt_0),139). orientation(lbbcgt_1,147,139). coords(lbbcgt_1,12,59,139).  visible(lbbcgt_1,139). happensAt(walking(lbbcgt_1),139).
orientation(lbbcgt_0,22,140). coords(lbbcgt_0,234,252,140).  visible(lbbcgt_0,140). happensAt(walking(lbbcgt_0),140). orientation(lbbcgt_1,147,140). coords(lbbcgt_1,12,60,140).  visible(lbbcgt_1,140). happensAt(walking(lbbcgt_1),140).
orientation(lbbcgt_0,22,141). coords(lbbcgt_0,235,256,141).  visible(lbbcgt_0,141). happensAt(walking(lbbcgt_0),141). orientation(lbbcgt_1,147,141). coords(lbbcgt_1,12,60,141).  visible(lbbcgt_1,141). happensAt(walking(lbbcgt_1),141).
orientation(lbbcgt_0,22,142). coords(lbbcgt_0,235,261,142).  visible(lbbcgt_0,142). happensAt(walking(lbbcgt_0),142). orientation(lbbcgt_1,147,142). coords(lbbcgt_1,11,61,142).  visible(lbbcgt_1,142). happensAt(walking(lbbcgt_1),142).
orientation(lbbcgt_0,22,143). coords(lbbcgt_0,235,265,143).  visible(lbbcgt_0,143). happensAt(walking(lbbcgt_0),143). orientation(lbbcgt_1,147,143). coords(lbbcgt_1,10,61,143).  visible(lbbcgt_1,143). happensAt(walking(lbbcgt_1),143).
orientation(lbbcgt_0,22,144). coords(lbbcgt_0,236,268,144).  visible(lbbcgt_0,144). happensAt(walking(lbbcgt_0),144). orientation(lbbcgt_1,147,144). coords(lbbcgt_1,10,62,144).  visible(lbbcgt_1,144). happensAt(walking(lbbcgt_1),144).
orientation(lbbcgt_0,22,145). coords(lbbcgt_0,236,270,145).  visible(lbbcgt_0,145). happensAt(walking(lbbcgt_0),145). orientation(lbbcgt_1,143,145). coords(lbbcgt_1,10,63,145).  visible(lbbcgt_1,145). happensAt(walking(lbbcgt_1),145).
orientation(lbbcgt_0,37,146). coords(lbbcgt_0,237,270,146).  visible(lbbcgt_0,146). happensAt(walking(lbbcgt_0),146). orientation(lbbcgt_1,143,146). coords(lbbcgt_1,9,64,146).  visible(lbbcgt_1,146). happensAt(walking(lbbcgt_1),146).
orientation(lbbcgt_0,37,147). coords(lbbcgt_0,239,270,147).  visible(lbbcgt_0,147). happensAt(walking(lbbcgt_0),147). orientation(lbbcgt_1,143,147). coords(lbbcgt_1,9,64,147).  visible(lbbcgt_1,147). happensAt(walking(lbbcgt_1),147).
orientation(lbbcgt_0,37,148). coords(lbbcgt_0,242,270,148).  visible(lbbcgt_0,148). happensAt(walking(lbbcgt_0),148). orientation(lbbcgt_1,143,148). coords(lbbcgt_1,9,65,148).  visible(lbbcgt_1,148). happensAt(walking(lbbcgt_1),148).
orientation(lbbcgt_0,37,149). coords(lbbcgt_0,242,270,149).  visible(lbbcgt_0,149). happensAt(walking(lbbcgt_0),149). orientation(lbbcgt_1,143,149). coords(lbbcgt_1,9,65,149).  visible(lbbcgt_1,149). happensAt(walking(lbbcgt_1),149).
orientation(lbbcgt_0,62,150). coords(lbbcgt_0,240,271,150).  visible(lbbcgt_0,150). happensAt(walking(lbbcgt_0),150). orientation(lbbcgt_1,143,150). coords(lbbcgt_1,9,68,150).  visible(lbbcgt_1,150). happensAt(walking(lbbcgt_1),150).
orientation(lbbcgt_0,62,151). coords(lbbcgt_0,242,272,151).  visible(lbbcgt_0,151). happensAt(walking(lbbcgt_0),151). orientation(lbbcgt_1,143,151). coords(lbbcgt_1,9,68,151).  visible(lbbcgt_1,151). happensAt(walking(lbbcgt_1),151).
orientation(lbbcgt_0,62,152). coords(lbbcgt_0,247,274,152).  visible(lbbcgt_0,152). happensAt(walking(lbbcgt_0),152). orientation(lbbcgt_1,143,152). coords(lbbcgt_1,8,68,152).  visible(lbbcgt_1,152). happensAt(walking(lbbcgt_1),152).
orientation(lbbcgt_0,62,153). coords(lbbcgt_0,251,276,153).  visible(lbbcgt_0,153). happensAt(walking(lbbcgt_0),153). orientation(lbbcgt_1,143,153). coords(lbbcgt_1,7,69,153).  visible(lbbcgt_1,153). happensAt(walking(lbbcgt_1),153).
orientation(lbbcgt_0,62,154). coords(lbbcgt_0,253,278,154).  visible(lbbcgt_0,154). happensAt(walking(lbbcgt_0),154). orientation(lbbcgt_1,143,154). coords(lbbcgt_1,7,69,154).  visible(lbbcgt_1,154). happensAt(walking(lbbcgt_1),154).
orientation(lbbcgt_0,62,155). coords(lbbcgt_0,261,281,155).  visible(lbbcgt_0,155). happensAt(walking(lbbcgt_0),155). orientation(lbbcgt_1,143,155). coords(lbbcgt_1,7,69,155).  visible(lbbcgt_1,155). happensAt(walking(lbbcgt_1),155).
orientation(lbbcgt_0,62,156). coords(lbbcgt_0,266,283,156).  visible(lbbcgt_0,156). happensAt(walking(lbbcgt_0),156). orientation(lbbcgt_1,143,156). coords(lbbcgt_1,7,71,156).  visible(lbbcgt_1,156). happensAt(walking(lbbcgt_1),156).
orientation(lbbcgt_0,62,157). coords(lbbcgt_0,266,286,157).  happensAt(disappear(lbbcgt_0),157). happensAt(walking(lbbcgt_0),157). orientation(lbbcgt_1,143,157). coords(lbbcgt_1,7,71,157).  visible(lbbcgt_1,157). happensAt(walking(lbbcgt_1),157).
#show holdsAt/2.