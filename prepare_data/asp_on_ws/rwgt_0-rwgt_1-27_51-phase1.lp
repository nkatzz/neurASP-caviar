
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

	orientation(rwgt_0,55,27). coords(rwgt_0,308,278,27).  visible(rwgt_0,27). happensAt(walking(rwgt_0),27). orientation(rwgt_1,96,27). coords(rwgt_1,269,285,27).  happensAt(appear(rwgt_1),27). happensAt(walking(rwgt_1),27).
orientation(rwgt_0,52,28). coords(rwgt_0,308,278,28).  visible(rwgt_0,28). happensAt(walking(rwgt_0),28). orientation(rwgt_1,109,28). coords(rwgt_1,267,283,28).  visible(rwgt_1,28). happensAt(walking(rwgt_1),28).
orientation(rwgt_0,55,29). coords(rwgt_0,308,278,29).  visible(rwgt_0,29). happensAt(walking(rwgt_0),29). orientation(rwgt_1,109,29). coords(rwgt_1,267,282,29).  visible(rwgt_1,29). happensAt(walking(rwgt_1),29).
orientation(rwgt_0,68,30). coords(rwgt_0,308,278,30).  visible(rwgt_0,30). happensAt(walking(rwgt_0),30). orientation(rwgt_1,109,30). coords(rwgt_1,267,281,30).  visible(rwgt_1,30). happensAt(walking(rwgt_1),30).
orientation(rwgt_0,67,31). coords(rwgt_0,308,278,31).  visible(rwgt_0,31). happensAt(walking(rwgt_0),31). orientation(rwgt_1,109,31). coords(rwgt_1,267,281,31).  visible(rwgt_1,31). happensAt(walking(rwgt_1),31).
orientation(rwgt_0,43,32). coords(rwgt_0,308,278,32).  visible(rwgt_0,32). happensAt(walking(rwgt_0),32). orientation(rwgt_1,109,32). coords(rwgt_1,267,281,32).  visible(rwgt_1,32). happensAt(walking(rwgt_1),32).
orientation(rwgt_0,42,33). coords(rwgt_0,308,278,33).  visible(rwgt_0,33). happensAt(walking(rwgt_0),33). orientation(rwgt_1,109,33). coords(rwgt_1,266,281,33).  visible(rwgt_1,33). happensAt(walking(rwgt_1),33).
orientation(rwgt_0,43,34). coords(rwgt_0,308,278,34).  visible(rwgt_0,34). happensAt(walking(rwgt_0),34). orientation(rwgt_1,109,34). coords(rwgt_1,265,281,34).  visible(rwgt_1,34). happensAt(walking(rwgt_1),34).
orientation(rwgt_0,51,35). coords(rwgt_0,307,278,35).  visible(rwgt_0,35). happensAt(walking(rwgt_0),35). orientation(rwgt_1,95,35). coords(rwgt_1,266,281,35).  visible(rwgt_1,35). happensAt(walking(rwgt_1),35).
orientation(rwgt_0,46,36). coords(rwgt_0,307,278,36).  visible(rwgt_0,36). happensAt(walking(rwgt_0),36). orientation(rwgt_1,72,36). coords(rwgt_1,266,281,36).  visible(rwgt_1,36). happensAt(walking(rwgt_1),36).
orientation(rwgt_0,37,37). coords(rwgt_0,307,278,37).  visible(rwgt_0,37). happensAt(walking(rwgt_0),37). orientation(rwgt_1,49,37). coords(rwgt_1,264,280,37).  visible(rwgt_1,37). happensAt(walking(rwgt_1),37).
orientation(rwgt_0,50,38). coords(rwgt_0,307,278,38).  visible(rwgt_0,38). happensAt(walking(rwgt_0),38). orientation(rwgt_1,40,38). coords(rwgt_1,260,279,38).  visible(rwgt_1,38). happensAt(walking(rwgt_1),38).
orientation(rwgt_0,42,39). coords(rwgt_0,307,278,39).  visible(rwgt_0,39). happensAt(walking(rwgt_0),39). orientation(rwgt_1,41,39). coords(rwgt_1,256,278,39).  visible(rwgt_1,39). happensAt(walking(rwgt_1),39).
orientation(rwgt_0,40,40). coords(rwgt_0,307,278,40).  visible(rwgt_0,40). happensAt(walking(rwgt_0),40). orientation(rwgt_1,35,40). coords(rwgt_1,253,275,40).  visible(rwgt_1,40). happensAt(walking(rwgt_1),40).
orientation(rwgt_0,46,41). coords(rwgt_0,307,278,41).  visible(rwgt_0,41). happensAt(walking(rwgt_0),41). orientation(rwgt_1,35,41). coords(rwgt_1,252,272,41).  visible(rwgt_1,41). happensAt(walking(rwgt_1),41).
orientation(rwgt_0,43,42). coords(rwgt_0,306,278,42).  visible(rwgt_0,42). happensAt(walking(rwgt_0),42). orientation(rwgt_1,35,42). coords(rwgt_1,251,269,42).  visible(rwgt_1,42). happensAt(walking(rwgt_1),42).
orientation(rwgt_0,38,43). coords(rwgt_0,306,278,43).  visible(rwgt_0,43). happensAt(walking(rwgt_0),43). orientation(rwgt_1,22,43). coords(rwgt_1,251,267,43).  visible(rwgt_1,43). happensAt(walking(rwgt_1),43).
orientation(rwgt_0,38,44). coords(rwgt_0,306,278,44).  visible(rwgt_0,44). happensAt(walking(rwgt_0),44). orientation(rwgt_1,22,44). coords(rwgt_1,250,266,44).  visible(rwgt_1,44). happensAt(walking(rwgt_1),44).
orientation(rwgt_0,40,45). coords(rwgt_0,305,278,45).  visible(rwgt_0,45). happensAt(walking(rwgt_0),45). orientation(rwgt_1,11,45). coords(rwgt_1,250,265,45).  visible(rwgt_1,45). happensAt(walking(rwgt_1),45).
orientation(rwgt_0,47,46). coords(rwgt_0,304,278,46).  visible(rwgt_0,46). happensAt(walking(rwgt_0),46). orientation(rwgt_1,11,46). coords(rwgt_1,248,264,46).  visible(rwgt_1,46). happensAt(walking(rwgt_1),46).
orientation(rwgt_0,44,47). coords(rwgt_0,304,278,47).  visible(rwgt_0,47). happensAt(walking(rwgt_0),47). orientation(rwgt_1,11,47). coords(rwgt_1,247,263,47).  visible(rwgt_1,47). happensAt(walking(rwgt_1),47).
orientation(rwgt_0,61,48). coords(rwgt_0,304,278,48).  visible(rwgt_0,48). happensAt(walking(rwgt_0),48). orientation(rwgt_1,3,48). coords(rwgt_1,246,262,48).  visible(rwgt_1,48). happensAt(walking(rwgt_1),48).
orientation(rwgt_0,53,49). coords(rwgt_0,304,278,49).  visible(rwgt_0,49). happensAt(walking(rwgt_0),49). orientation(rwgt_1,3,49). coords(rwgt_1,243,260,49).  visible(rwgt_1,49). happensAt(walking(rwgt_1),49).
orientation(rwgt_0,50,50). coords(rwgt_0,303,277,50).  visible(rwgt_0,50). happensAt(walking(rwgt_0),50). orientation(rwgt_1,3,50). coords(rwgt_1,242,258,50).  visible(rwgt_1,50). happensAt(walking(rwgt_1),50).
orientation(rwgt_0,49,51). coords(rwgt_0,303,276,51).  visible(rwgt_0,51). happensAt(walking(rwgt_0),51). orientation(rwgt_1,3,51). coords(rwgt_1,241,257,51).  visible(rwgt_1,51). happensAt(walking(rwgt_1),51).
#show holdsAt/2.