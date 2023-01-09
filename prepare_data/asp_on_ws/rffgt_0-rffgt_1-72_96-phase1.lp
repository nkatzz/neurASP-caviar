
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

	orientation(rffgt_0,134,72). coords(rffgt_0,134,61,72).  visible(rffgt_0,72). happensAt(active(rffgt_0),72). orientation(rffgt_1,134,72). coords(rffgt_1,155,113,72).  visible(rffgt_1,72). happensAt(active(rffgt_1),72).
orientation(rffgt_0,134,73). coords(rffgt_0,134,61,73).  visible(rffgt_0,73). happensAt(active(rffgt_0),73). orientation(rffgt_1,134,73). coords(rffgt_1,155,114,73).  visible(rffgt_1,73). happensAt(active(rffgt_1),73).
orientation(rffgt_0,134,74). coords(rffgt_0,134,61,74).  visible(rffgt_0,74). happensAt(active(rffgt_0),74). orientation(rffgt_1,134,74). coords(rffgt_1,155,114,74).  visible(rffgt_1,74). happensAt(active(rffgt_1),74).
orientation(rffgt_0,134,75). coords(rffgt_0,133,60,75).  visible(rffgt_0,75). happensAt(active(rffgt_0),75). orientation(rffgt_1,134,75). coords(rffgt_1,156,114,75).  visible(rffgt_1,75). happensAt(active(rffgt_1),75).
orientation(rffgt_0,134,76). coords(rffgt_0,133,59,76).  visible(rffgt_0,76). happensAt(active(rffgt_0),76). orientation(rffgt_1,134,76). coords(rffgt_1,155,114,76).  visible(rffgt_1,76). happensAt(active(rffgt_1),76).
orientation(rffgt_0,134,77). coords(rffgt_0,133,59,77).  visible(rffgt_0,77). happensAt(active(rffgt_0),77). orientation(rffgt_1,138,77). coords(rffgt_1,155,114,77).  visible(rffgt_1,77). happensAt(active(rffgt_1),77).
orientation(rffgt_0,134,78). coords(rffgt_0,133,59,78).  visible(rffgt_0,78). happensAt(active(rffgt_0),78). orientation(rffgt_1,141,78). coords(rffgt_1,155,114,78).  visible(rffgt_1,78). happensAt(active(rffgt_1),78).
orientation(rffgt_0,134,79). coords(rffgt_0,133,58,79).  visible(rffgt_0,79). happensAt(active(rffgt_0),79). orientation(rffgt_1,143,79). coords(rffgt_1,155,114,79).  visible(rffgt_1,79). happensAt(active(rffgt_1),79).
orientation(rffgt_0,135,80). coords(rffgt_0,133,56,80).  visible(rffgt_0,80). happensAt(walking(rffgt_0),80). orientation(rffgt_1,143,80). coords(rffgt_1,156,114,80).  visible(rffgt_1,80). happensAt(active(rffgt_1),80).
orientation(rffgt_0,136,81). coords(rffgt_0,133,56,81).  visible(rffgt_0,81). happensAt(walking(rffgt_0),81). orientation(rffgt_1,143,81). coords(rffgt_1,156,114,81).  visible(rffgt_1,81). happensAt(active(rffgt_1),81).
orientation(rffgt_0,137,82). coords(rffgt_0,133,55,82).  visible(rffgt_0,82). happensAt(walking(rffgt_0),82). orientation(rffgt_1,143,82). coords(rffgt_1,156,113,82).  visible(rffgt_1,82). happensAt(active(rffgt_1),82).
orientation(rffgt_0,136,83). coords(rffgt_0,134,54,83).  visible(rffgt_0,83). happensAt(walking(rffgt_0),83). orientation(rffgt_1,143,83). coords(rffgt_1,156,111,83).  visible(rffgt_1,83). happensAt(active(rffgt_1),83).
orientation(rffgt_0,138,84). coords(rffgt_0,135,53,84).  visible(rffgt_0,84). happensAt(walking(rffgt_0),84). orientation(rffgt_1,143,84). coords(rffgt_1,156,109,84).  visible(rffgt_1,84). happensAt(active(rffgt_1),84).
orientation(rffgt_0,137,85). coords(rffgt_0,136,53,85).  visible(rffgt_0,85). happensAt(walking(rffgt_0),85). orientation(rffgt_1,143,85). coords(rffgt_1,156,106,85).  visible(rffgt_1,85). happensAt(active(rffgt_1),85).
orientation(rffgt_0,139,86). coords(rffgt_0,137,53,86).  visible(rffgt_0,86). happensAt(walking(rffgt_0),86). orientation(rffgt_1,143,86). coords(rffgt_1,156,104,86).  visible(rffgt_1,86). happensAt(active(rffgt_1),86).
orientation(rffgt_0,140,87). coords(rffgt_0,137,52,87).  visible(rffgt_0,87). happensAt(walking(rffgt_0),87). orientation(rffgt_1,143,87). coords(rffgt_1,156,103,87).  visible(rffgt_1,87). happensAt(active(rffgt_1),87).
orientation(rffgt_0,140,88). coords(rffgt_0,137,52,88).  visible(rffgt_0,88). happensAt(walking(rffgt_0),88). orientation(rffgt_1,143,88). coords(rffgt_1,155,102,88).  visible(rffgt_1,88). happensAt(active(rffgt_1),88).
orientation(rffgt_0,140,89). coords(rffgt_0,137,52,89).  visible(rffgt_0,89). happensAt(walking(rffgt_0),89). orientation(rffgt_1,143,89). coords(rffgt_1,153,102,89).  visible(rffgt_1,89). happensAt(active(rffgt_1),89).
orientation(rffgt_0,140,90). coords(rffgt_0,137,51,90).  visible(rffgt_0,90). happensAt(walking(rffgt_0),90). orientation(rffgt_1,145,90). coords(rffgt_1,152,103,90).  visible(rffgt_1,90). happensAt(active(rffgt_1),90).
orientation(rffgt_0,135,91). coords(rffgt_0,138,51,91).  visible(rffgt_0,91). happensAt(walking(rffgt_0),91). orientation(rffgt_1,145,91). coords(rffgt_1,151,103,91).  visible(rffgt_1,91). happensAt(active(rffgt_1),91).
orientation(rffgt_0,135,92). coords(rffgt_0,138,51,92).  visible(rffgt_0,92). happensAt(walking(rffgt_0),92). orientation(rffgt_1,145,92). coords(rffgt_1,151,104,92).  visible(rffgt_1,92). happensAt(active(rffgt_1),92).
orientation(rffgt_0,135,93). coords(rffgt_0,138,49,93).  visible(rffgt_0,93). happensAt(walking(rffgt_0),93). orientation(rffgt_1,145,93). coords(rffgt_1,150,105,93).  visible(rffgt_1,93). happensAt(active(rffgt_1),93).
orientation(rffgt_0,135,94). coords(rffgt_0,138,49,94).  visible(rffgt_0,94). happensAt(walking(rffgt_0),94). orientation(rffgt_1,146,94). coords(rffgt_1,149,106,94).  visible(rffgt_1,94). happensAt(active(rffgt_1),94).
orientation(rffgt_0,135,95). coords(rffgt_0,138,47,95).  visible(rffgt_0,95). happensAt(walking(rffgt_0),95). orientation(rffgt_1,146,95). coords(rffgt_1,149,106,95).  visible(rffgt_1,95). happensAt(active(rffgt_1),95).
orientation(rffgt_0,135,96). coords(rffgt_0,139,46,96).  visible(rffgt_0,96). happensAt(walking(rffgt_0),96). orientation(rffgt_1,146,96). coords(rffgt_1,150,106,96).  visible(rffgt_1,96). happensAt(active(rffgt_1),96).
#show holdsAt/2.