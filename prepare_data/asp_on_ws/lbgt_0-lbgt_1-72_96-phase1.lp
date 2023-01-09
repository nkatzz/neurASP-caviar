
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

	orientation(lbgt_0,154,72). coords(lbgt_0,159,120,72).  visible(lbgt_0,72). happensAt(walking(lbgt_0),72). orientation(lbgt_1,145,72). coords(lbgt_1,86,65,72).  happensAt(appear(lbgt_1),72). happensAt(walking(lbgt_1),72).
orientation(lbgt_0,154,73). coords(lbgt_0,159,119,73).  visible(lbgt_0,73). happensAt(walking(lbgt_0),73). orientation(lbgt_1,145,73). coords(lbgt_1,86,65,73).  visible(lbgt_1,73). happensAt(walking(lbgt_1),73).
orientation(lbgt_0,154,74). coords(lbgt_0,159,118,74).  visible(lbgt_0,74). happensAt(walking(lbgt_0),74). orientation(lbgt_1,145,74). coords(lbgt_1,86,65,74).  visible(lbgt_1,74). happensAt(walking(lbgt_1),74).
orientation(lbgt_0,154,75). coords(lbgt_0,158,117,75).  visible(lbgt_0,75). happensAt(walking(lbgt_0),75). orientation(lbgt_1,145,75). coords(lbgt_1,86,65,75).  visible(lbgt_1,75). happensAt(walking(lbgt_1),75).
orientation(lbgt_0,154,76). coords(lbgt_0,157,115,76).  visible(lbgt_0,76). happensAt(walking(lbgt_0),76). orientation(lbgt_1,145,76). coords(lbgt_1,82,65,76).  visible(lbgt_1,76). happensAt(walking(lbgt_1),76).
orientation(lbgt_0,154,77). coords(lbgt_0,155,114,77).  visible(lbgt_0,77). happensAt(walking(lbgt_0),77). orientation(lbgt_1,145,77). coords(lbgt_1,82,66,77).  visible(lbgt_1,77). happensAt(walking(lbgt_1),77).
orientation(lbgt_0,154,78). coords(lbgt_0,155,113,78).  visible(lbgt_0,78). happensAt(walking(lbgt_0),78). orientation(lbgt_1,145,78). coords(lbgt_1,81,67,78).  visible(lbgt_1,78). happensAt(walking(lbgt_1),78).
orientation(lbgt_0,154,79). coords(lbgt_0,154,111,79).  visible(lbgt_0,79). happensAt(walking(lbgt_0),79). orientation(lbgt_1,145,79). coords(lbgt_1,79,68,79).  visible(lbgt_1,79). happensAt(walking(lbgt_1),79).
orientation(lbgt_0,154,80). coords(lbgt_0,153,110,80).  visible(lbgt_0,80). happensAt(walking(lbgt_0),80). orientation(lbgt_1,145,80). coords(lbgt_1,79,68,80).  visible(lbgt_1,80). happensAt(walking(lbgt_1),80).
orientation(lbgt_0,154,81). coords(lbgt_0,153,109,81).  visible(lbgt_0,81). happensAt(walking(lbgt_0),81). orientation(lbgt_1,145,81). coords(lbgt_1,79,68,81).  visible(lbgt_1,81). happensAt(walking(lbgt_1),81).
orientation(lbgt_0,154,82). coords(lbgt_0,152,109,82).  visible(lbgt_0,82). happensAt(walking(lbgt_0),82). orientation(lbgt_1,145,82). coords(lbgt_1,79,68,82).  visible(lbgt_1,82). happensAt(walking(lbgt_1),82).
orientation(lbgt_0,154,83). coords(lbgt_0,152,109,83).  visible(lbgt_0,83). happensAt(walking(lbgt_0),83). orientation(lbgt_1,145,83). coords(lbgt_1,79,68,83).  visible(lbgt_1,83). happensAt(walking(lbgt_1),83).
orientation(lbgt_0,154,84). coords(lbgt_0,152,108,84).  visible(lbgt_0,84). happensAt(walking(lbgt_0),84). orientation(lbgt_1,145,84). coords(lbgt_1,79,68,84).  visible(lbgt_1,84). happensAt(walking(lbgt_1),84).
orientation(lbgt_0,154,85). coords(lbgt_0,152,108,85).  visible(lbgt_0,85). happensAt(walking(lbgt_0),85). orientation(lbgt_1,145,85). coords(lbgt_1,78,68,85).  visible(lbgt_1,85). happensAt(walking(lbgt_1),85).
orientation(lbgt_0,154,86). coords(lbgt_0,152,108,86).  visible(lbgt_0,86). happensAt(walking(lbgt_0),86). orientation(lbgt_1,145,86). coords(lbgt_1,78,68,86).  visible(lbgt_1,86). happensAt(walking(lbgt_1),86).
orientation(lbgt_0,154,87). coords(lbgt_0,152,108,87).  visible(lbgt_0,87). happensAt(walking(lbgt_0),87). orientation(lbgt_1,145,87). coords(lbgt_1,78,68,87).  visible(lbgt_1,87). happensAt(walking(lbgt_1),87).
orientation(lbgt_0,154,88). coords(lbgt_0,152,108,88).  visible(lbgt_0,88). happensAt(walking(lbgt_0),88). orientation(lbgt_1,145,88). coords(lbgt_1,77,68,88).  visible(lbgt_1,88). happensAt(walking(lbgt_1),88).
orientation(lbgt_0,154,89). coords(lbgt_0,152,108,89).  visible(lbgt_0,89). happensAt(walking(lbgt_0),89). orientation(lbgt_1,145,89). coords(lbgt_1,77,68,89).  visible(lbgt_1,89). happensAt(walking(lbgt_1),89).
orientation(lbgt_0,154,90). coords(lbgt_0,151,108,90).  visible(lbgt_0,90). happensAt(walking(lbgt_0),90). orientation(lbgt_1,145,90). coords(lbgt_1,77,68,90).  visible(lbgt_1,90). happensAt(walking(lbgt_1),90).
orientation(lbgt_0,154,91). coords(lbgt_0,151,108,91).  visible(lbgt_0,91). happensAt(inactive(lbgt_0),91). orientation(lbgt_1,145,91). coords(lbgt_1,77,71,91).  visible(lbgt_1,91). happensAt(walking(lbgt_1),91).
orientation(lbgt_0,154,92). coords(lbgt_0,151,108,92).  visible(lbgt_0,92). happensAt(inactive(lbgt_0),92). orientation(lbgt_1,145,92). coords(lbgt_1,77,72,92).  visible(lbgt_1,92). happensAt(walking(lbgt_1),92).
orientation(lbgt_0,154,93). coords(lbgt_0,151,108,93).  visible(lbgt_0,93). happensAt(inactive(lbgt_0),93). orientation(lbgt_1,145,93). coords(lbgt_1,76,72,93).  visible(lbgt_1,93). happensAt(walking(lbgt_1),93).
orientation(lbgt_0,154,94). coords(lbgt_0,151,108,94).  visible(lbgt_0,94). happensAt(inactive(lbgt_0),94). orientation(lbgt_1,145,94). coords(lbgt_1,76,72,94).  visible(lbgt_1,94). happensAt(walking(lbgt_1),94).
orientation(lbgt_0,154,95). coords(lbgt_0,151,108,95).  visible(lbgt_0,95). happensAt(inactive(lbgt_0),95). orientation(lbgt_1,145,95). coords(lbgt_1,76,73,95).  visible(lbgt_1,95). happensAt(walking(lbgt_1),95).
orientation(lbgt_0,154,96). coords(lbgt_0,151,108,96).  visible(lbgt_0,96). happensAt(inactive(lbgt_0),96). orientation(lbgt_1,145,96). coords(lbgt_1,76,73,96).  visible(lbgt_1,96). happensAt(walking(lbgt_1),96).
#show holdsAt/2.