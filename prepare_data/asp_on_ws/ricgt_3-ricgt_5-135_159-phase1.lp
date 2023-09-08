
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

	orientation(ricgt_3,155,135). coords(ricgt_3,187,156,135).  visible(ricgt_3,135). happensAt(inactive(ricgt_3),135). orientation(ricgt_5,0,135). coords(ricgt_5,27,127,135).  visible(ricgt_5,135). happensAt(walking(ricgt_5),135).
orientation(ricgt_3,155,136). coords(ricgt_3,187,156,136).  visible(ricgt_3,136). happensAt(inactive(ricgt_3),136). orientation(ricgt_5,0,136). coords(ricgt_5,27,127,136).  visible(ricgt_5,136). happensAt(walking(ricgt_5),136).
orientation(ricgt_3,155,137). coords(ricgt_3,187,156,137).  visible(ricgt_3,137). happensAt(inactive(ricgt_3),137). orientation(ricgt_5,0,137). coords(ricgt_5,27,127,137).  visible(ricgt_5,137). happensAt(walking(ricgt_5),137).
orientation(ricgt_3,155,138). coords(ricgt_3,187,156,138).  visible(ricgt_3,138). happensAt(inactive(ricgt_3),138). orientation(ricgt_5,0,138). coords(ricgt_5,27,127,138).  visible(ricgt_5,138). happensAt(walking(ricgt_5),138).
orientation(ricgt_3,155,139). coords(ricgt_3,188,156,139).  visible(ricgt_3,139). happensAt(inactive(ricgt_3),139). orientation(ricgt_5,0,139). coords(ricgt_5,27,127,139).  visible(ricgt_5,139). happensAt(walking(ricgt_5),139).
orientation(ricgt_3,155,140). coords(ricgt_3,188,156,140).  visible(ricgt_3,140). happensAt(inactive(ricgt_3),140). orientation(ricgt_5,0,140). coords(ricgt_5,27,127,140).  visible(ricgt_5,140). happensAt(walking(ricgt_5),140).
orientation(ricgt_3,155,141). coords(ricgt_3,188,156,141).  visible(ricgt_3,141). happensAt(inactive(ricgt_3),141). orientation(ricgt_5,0,141). coords(ricgt_5,27,127,141).  visible(ricgt_5,141). happensAt(walking(ricgt_5),141).
orientation(ricgt_3,154,142). coords(ricgt_3,188,156,142).  visible(ricgt_3,142). happensAt(active(ricgt_3),142). orientation(ricgt_5,0,142). coords(ricgt_5,27,128,142).  visible(ricgt_5,142). happensAt(walking(ricgt_5),142).
orientation(ricgt_3,154,143). coords(ricgt_3,189,156,143).  visible(ricgt_3,143). happensAt(active(ricgt_3),143). orientation(ricgt_5,0,143). coords(ricgt_5,27,128,143).  visible(ricgt_5,143). happensAt(walking(ricgt_5),143).
orientation(ricgt_3,153,144). coords(ricgt_3,189,156,144).  visible(ricgt_3,144). happensAt(active(ricgt_3),144). orientation(ricgt_5,0,144). coords(ricgt_5,27,128,144).  visible(ricgt_5,144). happensAt(walking(ricgt_5),144).
orientation(ricgt_3,153,145). coords(ricgt_3,189,156,145).  visible(ricgt_3,145). happensAt(active(ricgt_3),145). orientation(ricgt_5,0,145). coords(ricgt_5,27,129,145).  visible(ricgt_5,145). happensAt(walking(ricgt_5),145).
orientation(ricgt_3,152,146). coords(ricgt_3,189,157,146).  visible(ricgt_3,146). happensAt(active(ricgt_3),146). orientation(ricgt_5,0,146). coords(ricgt_5,27,129,146).  visible(ricgt_5,146). happensAt(walking(ricgt_5),146).
orientation(ricgt_3,150,147). coords(ricgt_3,190,158,147).  visible(ricgt_3,147). happensAt(active(ricgt_3),147). orientation(ricgt_5,0,147). coords(ricgt_5,27,129,147).  visible(ricgt_5,147). happensAt(walking(ricgt_5),147).
orientation(ricgt_3,149,148). coords(ricgt_3,190,158,148).  visible(ricgt_3,148). happensAt(active(ricgt_3),148). orientation(ricgt_5,0,148). coords(ricgt_5,27,129,148).  visible(ricgt_5,148). happensAt(walking(ricgt_5),148).
orientation(ricgt_3,147,149). coords(ricgt_3,191,159,149).  visible(ricgt_3,149). happensAt(active(ricgt_3),149). orientation(ricgt_5,0,149). coords(ricgt_5,27,129,149).  visible(ricgt_5,149). happensAt(walking(ricgt_5),149).
orientation(ricgt_3,144,150). coords(ricgt_3,193,161,150).  visible(ricgt_3,150). happensAt(active(ricgt_3),150). orientation(ricgt_5,0,150). coords(ricgt_5,27,130,150).  visible(ricgt_5,150). happensAt(walking(ricgt_5),150).
orientation(ricgt_3,148,151). coords(ricgt_3,195,163,151).  visible(ricgt_3,151). happensAt(active(ricgt_3),151). orientation(ricgt_5,0,151). coords(ricgt_5,27,130,151).  visible(ricgt_5,151). happensAt(walking(ricgt_5),151).
orientation(ricgt_3,150,152). coords(ricgt_3,199,165,152).  visible(ricgt_3,152). happensAt(active(ricgt_3),152). orientation(ricgt_5,0,152). coords(ricgt_5,27,130,152).  visible(ricgt_5,152). happensAt(walking(ricgt_5),152).
orientation(ricgt_3,152,153). coords(ricgt_3,201,167,153).  visible(ricgt_3,153). happensAt(active(ricgt_3),153). orientation(ricgt_5,0,153). coords(ricgt_5,27,130,153).  visible(ricgt_5,153). happensAt(walking(ricgt_5),153).
orientation(ricgt_3,155,154). coords(ricgt_3,203,169,154).  visible(ricgt_3,154). happensAt(active(ricgt_3),154). orientation(ricgt_5,0,154). coords(ricgt_5,27,131,154).  visible(ricgt_5,154). happensAt(walking(ricgt_5),154).
orientation(ricgt_3,159,155). coords(ricgt_3,204,172,155).  visible(ricgt_3,155). happensAt(active(ricgt_3),155). orientation(ricgt_5,0,155). coords(ricgt_5,27,131,155).  visible(ricgt_5,155). happensAt(walking(ricgt_5),155).
orientation(ricgt_3,155,156). coords(ricgt_3,207,174,156).  visible(ricgt_3,156). happensAt(walking(ricgt_3),156). orientation(ricgt_5,0,156). coords(ricgt_5,27,132,156).  visible(ricgt_5,156). happensAt(walking(ricgt_5),156).
orientation(ricgt_3,154,157). coords(ricgt_3,207,175,157).  visible(ricgt_3,157). happensAt(walking(ricgt_3),157). orientation(ricgt_5,0,157). coords(ricgt_5,27,132,157).  visible(ricgt_5,157). happensAt(walking(ricgt_5),157).
orientation(ricgt_3,154,158). coords(ricgt_3,208,176,158).  visible(ricgt_3,158). happensAt(walking(ricgt_3),158). orientation(ricgt_5,0,158). coords(ricgt_5,27,132,158).  visible(ricgt_5,158). happensAt(walking(ricgt_5),158).
orientation(ricgt_3,159,159). coords(ricgt_3,209,178,159).  visible(ricgt_3,159). happensAt(walking(ricgt_3),159). orientation(ricgt_5,0,159). coords(ricgt_5,27,132,159).  visible(ricgt_5,159). happensAt(walking(ricgt_5),159).
#show holdsAt/2.