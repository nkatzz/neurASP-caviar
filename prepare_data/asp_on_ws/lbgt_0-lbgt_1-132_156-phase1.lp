
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

	orientation(lbgt_0,159,132). coords(lbgt_0,174,137,132).  visible(lbgt_0,132). happensAt(walking(lbgt_0),132). orientation(lbgt_1,155,132). coords(lbgt_1,76,86,132).  visible(lbgt_1,132). happensAt(walking(lbgt_1),132).
orientation(lbgt_0,170,133). coords(lbgt_0,177,138,133).  visible(lbgt_0,133). happensAt(walking(lbgt_0),133). orientation(lbgt_1,155,133). coords(lbgt_1,76,86,133).  visible(lbgt_1,133). happensAt(walking(lbgt_1),133).
orientation(lbgt_0,175,134). coords(lbgt_0,179,139,134).  visible(lbgt_0,134). happensAt(walking(lbgt_0),134). orientation(lbgt_1,155,134). coords(lbgt_1,76,88,134).  visible(lbgt_1,134). happensAt(walking(lbgt_1),134).
orientation(lbgt_0,175,135). coords(lbgt_0,182,141,135).  visible(lbgt_0,135). happensAt(walking(lbgt_0),135). orientation(lbgt_1,147,135). coords(lbgt_1,76,88,135).  visible(lbgt_1,135). happensAt(walking(lbgt_1),135).
orientation(lbgt_0,168,136). coords(lbgt_0,183,145,136).  visible(lbgt_0,136). happensAt(walking(lbgt_0),136). orientation(lbgt_1,147,136). coords(lbgt_1,77,88,136).  visible(lbgt_1,136). happensAt(walking(lbgt_1),136).
orientation(lbgt_0,168,137). coords(lbgt_0,185,148,137).  visible(lbgt_0,137). happensAt(walking(lbgt_0),137). orientation(lbgt_1,147,137). coords(lbgt_1,77,89,137).  visible(lbgt_1,137). happensAt(walking(lbgt_1),137).
orientation(lbgt_0,168,138). coords(lbgt_0,186,150,138).  visible(lbgt_0,138). happensAt(walking(lbgt_0),138). orientation(lbgt_1,147,138). coords(lbgt_1,77,89,138).  visible(lbgt_1,138). happensAt(walking(lbgt_1),138).
orientation(lbgt_0,168,139). coords(lbgt_0,188,152,139).  visible(lbgt_0,139). happensAt(walking(lbgt_0),139). orientation(lbgt_1,147,139). coords(lbgt_1,77,90,139).  visible(lbgt_1,139). happensAt(walking(lbgt_1),139).
orientation(lbgt_0,168,140). coords(lbgt_0,188,155,140).  visible(lbgt_0,140). happensAt(walking(lbgt_0),140). orientation(lbgt_1,153,140). coords(lbgt_1,77,89,140).  visible(lbgt_1,140). happensAt(walking(lbgt_1),140).
orientation(lbgt_0,168,141). coords(lbgt_0,189,157,141).  visible(lbgt_0,141). happensAt(walking(lbgt_0),141). orientation(lbgt_1,153,141). coords(lbgt_1,77,91,141).  visible(lbgt_1,141). happensAt(walking(lbgt_1),141).
orientation(lbgt_0,163,142). coords(lbgt_0,190,161,142).  visible(lbgt_0,142). happensAt(walking(lbgt_0),142). orientation(lbgt_1,153,142). coords(lbgt_1,77,92,142).  visible(lbgt_1,142). happensAt(walking(lbgt_1),142).
orientation(lbgt_0,163,143). coords(lbgt_0,189,165,143).  visible(lbgt_0,143). happensAt(walking(lbgt_0),143). orientation(lbgt_1,153,143). coords(lbgt_1,77,91,143).  visible(lbgt_1,143). happensAt(walking(lbgt_1),143).
orientation(lbgt_0,154,144). coords(lbgt_0,190,170,144).  visible(lbgt_0,144). happensAt(walking(lbgt_0),144). orientation(lbgt_1,153,144). coords(lbgt_1,77,92,144).  visible(lbgt_1,144). happensAt(walking(lbgt_1),144).
orientation(lbgt_0,154,145). coords(lbgt_0,192,174,145).  visible(lbgt_0,145). happensAt(walking(lbgt_0),145). orientation(lbgt_1,153,145). coords(lbgt_1,77,91,145).  visible(lbgt_1,145). happensAt(walking(lbgt_1),145).
orientation(lbgt_0,153,146). coords(lbgt_0,194,178,146).  visible(lbgt_0,146). happensAt(walking(lbgt_0),146). orientation(lbgt_1,153,146). coords(lbgt_1,77,92,146).  visible(lbgt_1,146). happensAt(walking(lbgt_1),146).
orientation(lbgt_0,153,147). coords(lbgt_0,195,180,147).  visible(lbgt_0,147). happensAt(walking(lbgt_0),147). orientation(lbgt_1,153,147). coords(lbgt_1,77,92,147).  visible(lbgt_1,147). happensAt(walking(lbgt_1),147).
orientation(lbgt_0,153,148). coords(lbgt_0,196,182,148).  visible(lbgt_0,148). happensAt(walking(lbgt_0),148). orientation(lbgt_1,153,148). coords(lbgt_1,77,92,148).  visible(lbgt_1,148). happensAt(walking(lbgt_1),148).
orientation(lbgt_0,153,149). coords(lbgt_0,197,184,149).  visible(lbgt_0,149). happensAt(walking(lbgt_0),149). orientation(lbgt_1,157,149). coords(lbgt_1,78,93,149).  visible(lbgt_1,149). happensAt(walking(lbgt_1),149).
orientation(lbgt_0,153,150). coords(lbgt_0,199,186,150).  visible(lbgt_0,150). happensAt(walking(lbgt_0),150). orientation(lbgt_1,157,150). coords(lbgt_1,78,93,150).  visible(lbgt_1,150). happensAt(walking(lbgt_1),150).
orientation(lbgt_0,153,151). coords(lbgt_0,201,187,151).  visible(lbgt_0,151). happensAt(walking(lbgt_0),151). orientation(lbgt_1,157,151). coords(lbgt_1,78,93,151).  visible(lbgt_1,151). happensAt(walking(lbgt_1),151).
orientation(lbgt_0,160,152). coords(lbgt_0,203,187,152).  visible(lbgt_0,152). happensAt(walking(lbgt_0),152). orientation(lbgt_1,157,152). coords(lbgt_1,78,93,152).  visible(lbgt_1,152). happensAt(walking(lbgt_1),152).
orientation(lbgt_0,166,153). coords(lbgt_0,206,189,153).  visible(lbgt_0,153). happensAt(walking(lbgt_0),153). orientation(lbgt_1,157,153). coords(lbgt_1,79,93,153).  visible(lbgt_1,153). happensAt(walking(lbgt_1),153).
orientation(lbgt_0,166,154). coords(lbgt_0,208,195,154).  visible(lbgt_0,154). happensAt(walking(lbgt_0),154). orientation(lbgt_1,157,154). coords(lbgt_1,79,94,154).  visible(lbgt_1,154). happensAt(walking(lbgt_1),154).
orientation(lbgt_0,161,155). coords(lbgt_0,209,198,155).  visible(lbgt_0,155). happensAt(walking(lbgt_0),155). orientation(lbgt_1,157,155). coords(lbgt_1,79,94,155).  visible(lbgt_1,155). happensAt(walking(lbgt_1),155).
orientation(lbgt_0,161,156). coords(lbgt_0,211,199,156).  visible(lbgt_0,156). happensAt(walking(lbgt_0),156). orientation(lbgt_1,157,156). coords(lbgt_1,79,94,156).  visible(lbgt_1,156). happensAt(walking(lbgt_1),156).
#show holdsAt/2.