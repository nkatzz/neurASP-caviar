
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

	orientation(ricgt_4,172,131). coords(ricgt_4,26,134,131).  visible(ricgt_4,131). happensAt(walking(ricgt_4),131). orientation(ricgt_5,0,131). coords(ricgt_5,27,127,131).  visible(ricgt_5,131). happensAt(walking(ricgt_5),131).
orientation(ricgt_4,172,132). coords(ricgt_4,26,134,132).  visible(ricgt_4,132). happensAt(walking(ricgt_4),132). orientation(ricgt_5,0,132). coords(ricgt_5,27,127,132).  visible(ricgt_5,132). happensAt(walking(ricgt_5),132).
orientation(ricgt_4,172,133). coords(ricgt_4,26,134,133).  visible(ricgt_4,133). happensAt(walking(ricgt_4),133). orientation(ricgt_5,0,133). coords(ricgt_5,27,127,133).  visible(ricgt_5,133). happensAt(walking(ricgt_5),133).
orientation(ricgt_4,172,134). coords(ricgt_4,26,134,134).  visible(ricgt_4,134). happensAt(walking(ricgt_4),134). orientation(ricgt_5,0,134). coords(ricgt_5,27,127,134).  visible(ricgt_5,134). happensAt(walking(ricgt_5),134).
orientation(ricgt_4,172,135). coords(ricgt_4,26,134,135).  visible(ricgt_4,135). happensAt(walking(ricgt_4),135). orientation(ricgt_5,0,135). coords(ricgt_5,27,127,135).  visible(ricgt_5,135). happensAt(walking(ricgt_5),135).
orientation(ricgt_4,172,136). coords(ricgt_4,26,135,136).  visible(ricgt_4,136). happensAt(walking(ricgt_4),136). orientation(ricgt_5,0,136). coords(ricgt_5,27,127,136).  visible(ricgt_5,136). happensAt(walking(ricgt_5),136).
orientation(ricgt_4,172,137). coords(ricgt_4,26,135,137).  visible(ricgt_4,137). happensAt(walking(ricgt_4),137). orientation(ricgt_5,0,137). coords(ricgt_5,27,127,137).  visible(ricgt_5,137). happensAt(walking(ricgt_5),137).
orientation(ricgt_4,172,138). coords(ricgt_4,26,135,138).  visible(ricgt_4,138). happensAt(walking(ricgt_4),138). orientation(ricgt_5,0,138). coords(ricgt_5,27,127,138).  visible(ricgt_5,138). happensAt(walking(ricgt_5),138).
orientation(ricgt_4,172,139). coords(ricgt_4,26,136,139).  visible(ricgt_4,139). happensAt(walking(ricgt_4),139). orientation(ricgt_5,0,139). coords(ricgt_5,27,127,139).  visible(ricgt_5,139). happensAt(walking(ricgt_5),139).
orientation(ricgt_4,172,140). coords(ricgt_4,26,136,140).  visible(ricgt_4,140). happensAt(walking(ricgt_4),140). orientation(ricgt_5,0,140). coords(ricgt_5,27,127,140).  visible(ricgt_5,140). happensAt(walking(ricgt_5),140).
orientation(ricgt_4,172,141). coords(ricgt_4,26,136,141).  visible(ricgt_4,141). happensAt(walking(ricgt_4),141). orientation(ricgt_5,0,141). coords(ricgt_5,27,127,141).  visible(ricgt_5,141). happensAt(walking(ricgt_5),141).
orientation(ricgt_4,172,142). coords(ricgt_4,26,136,142).  visible(ricgt_4,142). happensAt(walking(ricgt_4),142). orientation(ricgt_5,0,142). coords(ricgt_5,27,128,142).  visible(ricgt_5,142). happensAt(walking(ricgt_5),142).
orientation(ricgt_4,172,143). coords(ricgt_4,26,136,143).  visible(ricgt_4,143). happensAt(walking(ricgt_4),143). orientation(ricgt_5,0,143). coords(ricgt_5,27,128,143).  visible(ricgt_5,143). happensAt(walking(ricgt_5),143).
orientation(ricgt_4,172,144). coords(ricgt_4,26,137,144).  visible(ricgt_4,144). happensAt(walking(ricgt_4),144). orientation(ricgt_5,0,144). coords(ricgt_5,27,128,144).  visible(ricgt_5,144). happensAt(walking(ricgt_5),144).
orientation(ricgt_4,172,145). coords(ricgt_4,25,137,145).  visible(ricgt_4,145). happensAt(walking(ricgt_4),145). orientation(ricgt_5,0,145). coords(ricgt_5,27,129,145).  visible(ricgt_5,145). happensAt(walking(ricgt_5),145).
orientation(ricgt_4,172,146). coords(ricgt_4,25,137,146).  visible(ricgt_4,146). happensAt(walking(ricgt_4),146). orientation(ricgt_5,0,146). coords(ricgt_5,27,129,146).  visible(ricgt_5,146). happensAt(walking(ricgt_5),146).
orientation(ricgt_4,172,147). coords(ricgt_4,25,137,147).  visible(ricgt_4,147). happensAt(walking(ricgt_4),147). orientation(ricgt_5,0,147). coords(ricgt_5,27,129,147).  visible(ricgt_5,147). happensAt(walking(ricgt_5),147).
orientation(ricgt_4,172,148). coords(ricgt_4,25,137,148).  visible(ricgt_4,148). happensAt(walking(ricgt_4),148). orientation(ricgt_5,0,148). coords(ricgt_5,27,129,148).  visible(ricgt_5,148). happensAt(walking(ricgt_5),148).
orientation(ricgt_4,172,149). coords(ricgt_4,25,137,149).  visible(ricgt_4,149). happensAt(walking(ricgt_4),149). orientation(ricgt_5,0,149). coords(ricgt_5,27,129,149).  visible(ricgt_5,149). happensAt(walking(ricgt_5),149).
orientation(ricgt_4,172,150). coords(ricgt_4,25,138,150).  visible(ricgt_4,150). happensAt(walking(ricgt_4),150). orientation(ricgt_5,0,150). coords(ricgt_5,27,130,150).  visible(ricgt_5,150). happensAt(walking(ricgt_5),150).
orientation(ricgt_4,172,151). coords(ricgt_4,25,138,151).  visible(ricgt_4,151). happensAt(walking(ricgt_4),151). orientation(ricgt_5,0,151). coords(ricgt_5,27,130,151).  visible(ricgt_5,151). happensAt(walking(ricgt_5),151).
orientation(ricgt_4,172,152). coords(ricgt_4,25,138,152).  visible(ricgt_4,152). happensAt(walking(ricgt_4),152). orientation(ricgt_5,0,152). coords(ricgt_5,27,130,152).  visible(ricgt_5,152). happensAt(walking(ricgt_5),152).
orientation(ricgt_4,172,153). coords(ricgt_4,25,138,153).  visible(ricgt_4,153). happensAt(walking(ricgt_4),153). orientation(ricgt_5,0,153). coords(ricgt_5,27,130,153).  visible(ricgt_5,153). happensAt(walking(ricgt_5),153).
orientation(ricgt_4,172,154). coords(ricgt_4,25,138,154).  visible(ricgt_4,154). happensAt(walking(ricgt_4),154). orientation(ricgt_5,0,154). coords(ricgt_5,27,131,154).  visible(ricgt_5,154). happensAt(walking(ricgt_5),154).
orientation(ricgt_4,172,155). coords(ricgt_4,25,138,155).  visible(ricgt_4,155). happensAt(walking(ricgt_4),155). orientation(ricgt_5,0,155). coords(ricgt_5,27,131,155).  visible(ricgt_5,155). happensAt(walking(ricgt_5),155).
#show holdsAt/2.