
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

	orientation(lb1gt_0,148,72). coords(lb1gt_0,175,113,72).  visible(lb1gt_0,72). happensAt(active(lb1gt_0),72). orientation(lb1gt_1,147,72). coords(lb1gt_1,74,104,72).  visible(lb1gt_1,72). happensAt(walking(lb1gt_1),72).
orientation(lb1gt_0,148,73). coords(lb1gt_0,173,112,73).  visible(lb1gt_0,73). happensAt(active(lb1gt_0),73). orientation(lb1gt_1,147,73). coords(lb1gt_1,75,104,73).  visible(lb1gt_1,73). happensAt(walking(lb1gt_1),73).
orientation(lb1gt_0,148,74). coords(lb1gt_0,172,112,74).  visible(lb1gt_0,74). happensAt(active(lb1gt_0),74). orientation(lb1gt_1,147,74). coords(lb1gt_1,75,105,74).  visible(lb1gt_1,74). happensAt(walking(lb1gt_1),74).
orientation(lb1gt_0,145,75). coords(lb1gt_0,170,111,75).  visible(lb1gt_0,75). happensAt(active(lb1gt_0),75). orientation(lb1gt_1,147,75). coords(lb1gt_1,75,105,75).  visible(lb1gt_1,75). happensAt(walking(lb1gt_1),75).
orientation(lb1gt_0,145,76). coords(lb1gt_0,169,110,76).  visible(lb1gt_0,76). happensAt(active(lb1gt_0),76). orientation(lb1gt_1,147,76). coords(lb1gt_1,75,106,76).  visible(lb1gt_1,76). happensAt(walking(lb1gt_1),76).
orientation(lb1gt_0,145,77). coords(lb1gt_0,168,110,77).  visible(lb1gt_0,77). happensAt(active(lb1gt_0),77). orientation(lb1gt_1,147,77). coords(lb1gt_1,75,107,77).  visible(lb1gt_1,77). happensAt(walking(lb1gt_1),77).
orientation(lb1gt_0,145,78). coords(lb1gt_0,167,110,78).  visible(lb1gt_0,78). happensAt(active(lb1gt_0),78). orientation(lb1gt_1,147,78). coords(lb1gt_1,75,107,78).  visible(lb1gt_1,78). happensAt(walking(lb1gt_1),78).
orientation(lb1gt_0,145,79). coords(lb1gt_0,167,109,79).  visible(lb1gt_0,79). happensAt(active(lb1gt_0),79). orientation(lb1gt_1,147,79). coords(lb1gt_1,75,108,79).  visible(lb1gt_1,79). happensAt(walking(lb1gt_1),79).
orientation(lb1gt_0,145,80). coords(lb1gt_0,167,109,80).  visible(lb1gt_0,80). happensAt(active(lb1gt_0),80). orientation(lb1gt_1,147,80). coords(lb1gt_1,75,109,80).  visible(lb1gt_1,80). happensAt(walking(lb1gt_1),80).
orientation(lb1gt_0,145,81). coords(lb1gt_0,167,109,81).  visible(lb1gt_0,81). happensAt(active(lb1gt_0),81). orientation(lb1gt_1,147,81). coords(lb1gt_1,75,109,81).  visible(lb1gt_1,81). happensAt(walking(lb1gt_1),81).
orientation(lb1gt_0,145,82). coords(lb1gt_0,166,109,82).  visible(lb1gt_0,82). happensAt(active(lb1gt_0),82). orientation(lb1gt_1,147,82). coords(lb1gt_1,75,109,82).  visible(lb1gt_1,82). happensAt(walking(lb1gt_1),82).
orientation(lb1gt_0,145,83). coords(lb1gt_0,166,108,83).  visible(lb1gt_0,83). happensAt(active(lb1gt_0),83). orientation(lb1gt_1,147,83). coords(lb1gt_1,75,109,83).  visible(lb1gt_1,83). happensAt(walking(lb1gt_1),83).
orientation(lb1gt_0,145,84). coords(lb1gt_0,166,108,84).  visible(lb1gt_0,84). happensAt(active(lb1gt_0),84). orientation(lb1gt_1,147,84). coords(lb1gt_1,75,109,84).  visible(lb1gt_1,84). happensAt(walking(lb1gt_1),84).
orientation(lb1gt_0,145,85). coords(lb1gt_0,166,108,85).  visible(lb1gt_0,85). happensAt(active(lb1gt_0),85). orientation(lb1gt_1,147,85). coords(lb1gt_1,75,109,85).  visible(lb1gt_1,85). happensAt(walking(lb1gt_1),85).
orientation(lb1gt_0,145,86). coords(lb1gt_0,166,108,86).  visible(lb1gt_0,86). happensAt(active(lb1gt_0),86). orientation(lb1gt_1,147,86). coords(lb1gt_1,75,109,86).  visible(lb1gt_1,86). happensAt(walking(lb1gt_1),86).
orientation(lb1gt_0,145,87). coords(lb1gt_0,166,108,87).  visible(lb1gt_0,87). happensAt(active(lb1gt_0),87). orientation(lb1gt_1,147,87). coords(lb1gt_1,75,110,87).  visible(lb1gt_1,87). happensAt(walking(lb1gt_1),87).
orientation(lb1gt_0,145,88). coords(lb1gt_0,166,108,88).  visible(lb1gt_0,88). happensAt(active(lb1gt_0),88). orientation(lb1gt_1,147,88). coords(lb1gt_1,75,110,88).  visible(lb1gt_1,88). happensAt(walking(lb1gt_1),88).
orientation(lb1gt_0,145,89). coords(lb1gt_0,166,108,89).  visible(lb1gt_0,89). happensAt(active(lb1gt_0),89). orientation(lb1gt_1,147,89). coords(lb1gt_1,76,111,89).  visible(lb1gt_1,89). happensAt(walking(lb1gt_1),89).
orientation(lb1gt_0,145,90). coords(lb1gt_0,166,108,90).  visible(lb1gt_0,90). happensAt(active(lb1gt_0),90). orientation(lb1gt_1,147,90). coords(lb1gt_1,76,113,90).  visible(lb1gt_1,90). happensAt(walking(lb1gt_1),90).
orientation(lb1gt_0,145,91). coords(lb1gt_0,166,108,91).  visible(lb1gt_0,91). happensAt(active(lb1gt_0),91). orientation(lb1gt_1,147,91). coords(lb1gt_1,76,114,91).  visible(lb1gt_1,91). happensAt(walking(lb1gt_1),91).
orientation(lb1gt_0,145,92). coords(lb1gt_0,166,108,92).  visible(lb1gt_0,92). happensAt(active(lb1gt_0),92). orientation(lb1gt_1,147,92). coords(lb1gt_1,76,115,92).  visible(lb1gt_1,92). happensAt(walking(lb1gt_1),92).
orientation(lb1gt_0,145,93). coords(lb1gt_0,166,108,93).  visible(lb1gt_0,93). happensAt(active(lb1gt_0),93). orientation(lb1gt_1,147,93). coords(lb1gt_1,76,116,93).  visible(lb1gt_1,93). happensAt(walking(lb1gt_1),93).
orientation(lb1gt_0,145,94). coords(lb1gt_0,166,108,94).  visible(lb1gt_0,94). happensAt(active(lb1gt_0),94). orientation(lb1gt_1,148,94). coords(lb1gt_1,77,116,94).  visible(lb1gt_1,94). happensAt(walking(lb1gt_1),94).
orientation(lb1gt_0,145,95). coords(lb1gt_0,167,108,95).  visible(lb1gt_0,95). happensAt(active(lb1gt_0),95). orientation(lb1gt_1,148,95). coords(lb1gt_1,77,116,95).  visible(lb1gt_1,95). happensAt(walking(lb1gt_1),95).
orientation(lb1gt_0,145,96). coords(lb1gt_0,167,109,96).  visible(lb1gt_0,96). happensAt(active(lb1gt_0),96). orientation(lb1gt_1,148,96). coords(lb1gt_1,77,117,96).  visible(lb1gt_1,96). happensAt(walking(lb1gt_1),96).
#show holdsAt/2.