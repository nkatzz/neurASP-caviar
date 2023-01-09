
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

	orientation(wk1gt_0,165,135). coords(wk1gt_0,229,156,135).  visible(wk1gt_0,135). happensAt(walking(wk1gt_0),135). orientation(wk1gt_5,169,135). coords(wk1gt_5,44,169,135).  visible(wk1gt_5,135). happensAt(walking(wk1gt_5),135).
orientation(wk1gt_0,165,136). coords(wk1gt_0,229,158,136).  visible(wk1gt_0,136). happensAt(walking(wk1gt_0),136). orientation(wk1gt_5,169,136). coords(wk1gt_5,44,169,136).  visible(wk1gt_5,136). happensAt(walking(wk1gt_5),136).
orientation(wk1gt_0,165,137). coords(wk1gt_0,232,159,137).  visible(wk1gt_0,137). happensAt(walking(wk1gt_0),137). orientation(wk1gt_5,169,137). coords(wk1gt_5,44,169,137).  visible(wk1gt_5,137). happensAt(walking(wk1gt_5),137).
orientation(wk1gt_0,178,138). coords(wk1gt_0,234,162,138).  visible(wk1gt_0,138). happensAt(walking(wk1gt_0),138). orientation(wk1gt_5,169,138). coords(wk1gt_5,44,169,138).  visible(wk1gt_5,138). happensAt(walking(wk1gt_5),138).
orientation(wk1gt_0,7,139). coords(wk1gt_0,237,164,139).  visible(wk1gt_0,139). happensAt(walking(wk1gt_0),139). orientation(wk1gt_5,169,139). coords(wk1gt_5,44,169,139).  visible(wk1gt_5,139). happensAt(walking(wk1gt_5),139).
orientation(wk1gt_0,10,140). coords(wk1gt_0,240,167,140).  visible(wk1gt_0,140). happensAt(walking(wk1gt_0),140). orientation(wk1gt_5,169,140). coords(wk1gt_5,44,168,140).  visible(wk1gt_5,140). happensAt(walking(wk1gt_5),140).
orientation(wk1gt_0,2,141). coords(wk1gt_0,242,172,141).  visible(wk1gt_0,141). happensAt(walking(wk1gt_0),141). orientation(wk1gt_5,169,141). coords(wk1gt_5,45,168,141).  visible(wk1gt_5,141). happensAt(walking(wk1gt_5),141).
orientation(wk1gt_0,171,142). coords(wk1gt_0,244,178,142).  visible(wk1gt_0,142). happensAt(walking(wk1gt_0),142). orientation(wk1gt_5,169,142). coords(wk1gt_5,45,167,142).  visible(wk1gt_5,142). happensAt(walking(wk1gt_5),142).
orientation(wk1gt_0,170,143). coords(wk1gt_0,245,181,143).  visible(wk1gt_0,143). happensAt(walking(wk1gt_0),143). orientation(wk1gt_5,169,143). coords(wk1gt_5,45,167,143).  visible(wk1gt_5,143). happensAt(walking(wk1gt_5),143).
orientation(wk1gt_0,167,144). coords(wk1gt_0,247,185,144).  visible(wk1gt_0,144). happensAt(walking(wk1gt_0),144). orientation(wk1gt_5,169,144). coords(wk1gt_5,45,167,144).  visible(wk1gt_5,144). happensAt(walking(wk1gt_5),144).
orientation(wk1gt_0,175,145). coords(wk1gt_0,248,188,145).  visible(wk1gt_0,145). happensAt(walking(wk1gt_0),145). orientation(wk1gt_5,169,145). coords(wk1gt_5,45,167,145).  visible(wk1gt_5,145). happensAt(walking(wk1gt_5),145).
orientation(wk1gt_0,175,146). coords(wk1gt_0,249,191,146).  visible(wk1gt_0,146). happensAt(walking(wk1gt_0),146). orientation(wk1gt_5,162,146). coords(wk1gt_5,45,167,146).  visible(wk1gt_5,146). happensAt(walking(wk1gt_5),146).
orientation(wk1gt_0,179,147). coords(wk1gt_0,251,195,147).  visible(wk1gt_0,147). happensAt(walking(wk1gt_0),147). orientation(wk1gt_5,162,147). coords(wk1gt_5,45,167,147).  visible(wk1gt_5,147). happensAt(walking(wk1gt_5),147).
orientation(wk1gt_0,178,148). coords(wk1gt_0,253,201,148).  visible(wk1gt_0,148). happensAt(walking(wk1gt_0),148). orientation(wk1gt_5,162,148). coords(wk1gt_5,45,167,148).  visible(wk1gt_5,148). happensAt(walking(wk1gt_5),148).
orientation(wk1gt_0,178,149). coords(wk1gt_0,253,205,149).  visible(wk1gt_0,149). happensAt(walking(wk1gt_0),149). orientation(wk1gt_5,162,149). coords(wk1gt_5,45,167,149).  visible(wk1gt_5,149). happensAt(walking(wk1gt_5),149).
orientation(wk1gt_0,0,150). coords(wk1gt_0,253,209,150).  visible(wk1gt_0,150). happensAt(walking(wk1gt_0),150). orientation(wk1gt_5,162,150). coords(wk1gt_5,45,167,150).  visible(wk1gt_5,150). happensAt(walking(wk1gt_5),150).
orientation(wk1gt_0,0,151). coords(wk1gt_0,254,213,151).  visible(wk1gt_0,151). happensAt(walking(wk1gt_0),151). orientation(wk1gt_5,162,151). coords(wk1gt_5,45,167,151).  visible(wk1gt_5,151). happensAt(walking(wk1gt_5),151).
orientation(wk1gt_0,0,152). coords(wk1gt_0,255,217,152).  visible(wk1gt_0,152). happensAt(walking(wk1gt_0),152). orientation(wk1gt_5,162,152). coords(wk1gt_5,45,167,152).  visible(wk1gt_5,152). happensAt(walking(wk1gt_5),152).
orientation(wk1gt_0,0,153). coords(wk1gt_0,256,219,153).  visible(wk1gt_0,153). happensAt(walking(wk1gt_0),153). orientation(wk1gt_5,162,153). coords(wk1gt_5,45,167,153).  visible(wk1gt_5,153). happensAt(walking(wk1gt_5),153).
orientation(wk1gt_0,12,154). coords(wk1gt_0,257,221,154).  visible(wk1gt_0,154). happensAt(walking(wk1gt_0),154). orientation(wk1gt_5,162,154). coords(wk1gt_5,45,167,154).  visible(wk1gt_5,154). happensAt(walking(wk1gt_5),154).
orientation(wk1gt_0,28,155). coords(wk1gt_0,260,223,155).  visible(wk1gt_0,155). happensAt(walking(wk1gt_0),155). orientation(wk1gt_5,162,155). coords(wk1gt_5,45,167,155).  visible(wk1gt_5,155). happensAt(active(wk1gt_5),155).
orientation(wk1gt_0,25,156). coords(wk1gt_0,262,227,156).  visible(wk1gt_0,156). happensAt(walking(wk1gt_0),156). orientation(wk1gt_5,162,156). coords(wk1gt_5,46,166,156).  visible(wk1gt_5,156). happensAt(active(wk1gt_5),156).
orientation(wk1gt_0,22,157). coords(wk1gt_0,264,234,157).  visible(wk1gt_0,157). happensAt(walking(wk1gt_0),157). orientation(wk1gt_5,162,157). coords(wk1gt_5,46,166,157).  visible(wk1gt_5,157). happensAt(active(wk1gt_5),157).
orientation(wk1gt_0,12,158). coords(wk1gt_0,266,240,158).  visible(wk1gt_0,158). happensAt(walking(wk1gt_0),158). orientation(wk1gt_5,162,158). coords(wk1gt_5,46,166,158).  visible(wk1gt_5,158). happensAt(active(wk1gt_5),158).
orientation(wk1gt_0,23,159). coords(wk1gt_0,268,244,159).  visible(wk1gt_0,159). happensAt(walking(wk1gt_0),159). orientation(wk1gt_5,162,159). coords(wk1gt_5,46,166,159).  visible(wk1gt_5,159). happensAt(active(wk1gt_5),159).
#show holdsAt/2.