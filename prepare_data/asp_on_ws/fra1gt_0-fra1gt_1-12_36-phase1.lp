
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

	orientation(fra1gt_0,145,12). coords(fra1gt_0,190,143,12).  visible(fra1gt_0,12). happensAt(inactive(fra1gt_0),12). orientation(fra1gt_1,112,12). coords(fra1gt_1,277,20,12).  visible(fra1gt_1,12). happensAt(walking(fra1gt_1),12).
orientation(fra1gt_0,145,13). coords(fra1gt_0,190,143,13).  visible(fra1gt_0,13). happensAt(inactive(fra1gt_0),13). orientation(fra1gt_1,112,13). coords(fra1gt_1,277,20,13).  visible(fra1gt_1,13). happensAt(walking(fra1gt_1),13).
orientation(fra1gt_0,145,14). coords(fra1gt_0,191,143,14).  visible(fra1gt_0,14). happensAt(inactive(fra1gt_0),14). orientation(fra1gt_1,112,14). coords(fra1gt_1,277,20,14).  visible(fra1gt_1,14). happensAt(walking(fra1gt_1),14).
orientation(fra1gt_0,145,15). coords(fra1gt_0,191,143,15).  visible(fra1gt_0,15). happensAt(inactive(fra1gt_0),15). orientation(fra1gt_1,112,15). coords(fra1gt_1,277,20,15).  visible(fra1gt_1,15). happensAt(walking(fra1gt_1),15).
orientation(fra1gt_0,145,16). coords(fra1gt_0,191,143,16).  visible(fra1gt_0,16). happensAt(inactive(fra1gt_0),16). orientation(fra1gt_1,112,16). coords(fra1gt_1,277,20,16).  visible(fra1gt_1,16). happensAt(walking(fra1gt_1),16).
orientation(fra1gt_0,145,17). coords(fra1gt_0,191,143,17).  visible(fra1gt_0,17). happensAt(inactive(fra1gt_0),17). orientation(fra1gt_1,112,17). coords(fra1gt_1,277,20,17).  visible(fra1gt_1,17). happensAt(walking(fra1gt_1),17).
orientation(fra1gt_0,145,18). coords(fra1gt_0,191,144,18).  visible(fra1gt_0,18). happensAt(inactive(fra1gt_0),18). orientation(fra1gt_1,112,18). coords(fra1gt_1,277,19,18).  visible(fra1gt_1,18). happensAt(walking(fra1gt_1),18).
orientation(fra1gt_0,145,19). coords(fra1gt_0,191,144,19).  visible(fra1gt_0,19). happensAt(inactive(fra1gt_0),19). orientation(fra1gt_1,112,19). coords(fra1gt_1,277,19,19).  visible(fra1gt_1,19). happensAt(walking(fra1gt_1),19).
orientation(fra1gt_0,145,20). coords(fra1gt_0,191,144,20).  visible(fra1gt_0,20). happensAt(inactive(fra1gt_0),20). orientation(fra1gt_1,112,20). coords(fra1gt_1,277,18,20).  visible(fra1gt_1,20). happensAt(walking(fra1gt_1),20).
orientation(fra1gt_0,145,21). coords(fra1gt_0,191,145,21).  visible(fra1gt_0,21). happensAt(inactive(fra1gt_0),21). orientation(fra1gt_1,112,21). coords(fra1gt_1,277,16,21).  visible(fra1gt_1,21). happensAt(walking(fra1gt_1),21).
orientation(fra1gt_0,145,22). coords(fra1gt_0,193,146,22).  visible(fra1gt_0,22). happensAt(walking(fra1gt_0),22). orientation(fra1gt_1,108,22). coords(fra1gt_1,277,16,22).  visible(fra1gt_1,22). happensAt(walking(fra1gt_1),22).
orientation(fra1gt_0,145,23). coords(fra1gt_0,193,148,23).  visible(fra1gt_0,23). happensAt(walking(fra1gt_0),23). orientation(fra1gt_1,106,23). coords(fra1gt_1,277,14,23).  visible(fra1gt_1,23). happensAt(walking(fra1gt_1),23).
orientation(fra1gt_0,145,24). coords(fra1gt_0,195,149,24).  visible(fra1gt_0,24). happensAt(walking(fra1gt_0),24). orientation(fra1gt_1,106,24). coords(fra1gt_1,278,12,24).  visible(fra1gt_1,24). happensAt(walking(fra1gt_1),24).
orientation(fra1gt_0,145,25). coords(fra1gt_0,196,153,25).  visible(fra1gt_0,25). happensAt(walking(fra1gt_0),25). orientation(fra1gt_1,106,25). coords(fra1gt_1,278,12,25).  visible(fra1gt_1,25). happensAt(walking(fra1gt_1),25).
orientation(fra1gt_0,145,26). coords(fra1gt_0,199,154,26).  visible(fra1gt_0,26). happensAt(walking(fra1gt_0),26). orientation(fra1gt_1,106,26). coords(fra1gt_1,278,11,26).  visible(fra1gt_1,26). happensAt(walking(fra1gt_1),26).
orientation(fra1gt_0,145,27). coords(fra1gt_0,201,156,27).  visible(fra1gt_0,27). happensAt(walking(fra1gt_0),27). orientation(fra1gt_1,106,27). coords(fra1gt_1,278,11,27).  visible(fra1gt_1,27). happensAt(walking(fra1gt_1),27).
orientation(fra1gt_0,145,28). coords(fra1gt_0,204,158,28).  visible(fra1gt_0,28). happensAt(walking(fra1gt_0),28). orientation(fra1gt_1,106,28). coords(fra1gt_1,278,11,28).  visible(fra1gt_1,28). happensAt(walking(fra1gt_1),28).
orientation(fra1gt_0,145,29). coords(fra1gt_0,207,159,29).  visible(fra1gt_0,29). happensAt(walking(fra1gt_0),29). orientation(fra1gt_1,106,29). coords(fra1gt_1,278,11,29).  visible(fra1gt_1,29). happensAt(walking(fra1gt_1),29).
orientation(fra1gt_0,145,30). coords(fra1gt_0,208,162,30).  visible(fra1gt_0,30). happensAt(walking(fra1gt_0),30). orientation(fra1gt_1,106,30). coords(fra1gt_1,278,11,30).  visible(fra1gt_1,30). happensAt(walking(fra1gt_1),30).
orientation(fra1gt_0,145,31). coords(fra1gt_0,209,163,31).  visible(fra1gt_0,31). happensAt(walking(fra1gt_0),31). orientation(fra1gt_1,106,31). coords(fra1gt_1,278,11,31).  visible(fra1gt_1,31). happensAt(walking(fra1gt_1),31).
orientation(fra1gt_0,145,32). coords(fra1gt_0,209,165,32).  visible(fra1gt_0,32). happensAt(walking(fra1gt_0),32). orientation(fra1gt_1,103,32). coords(fra1gt_1,278,11,32).  visible(fra1gt_1,32). happensAt(walking(fra1gt_1),32).
orientation(fra1gt_0,156,33). coords(fra1gt_0,209,167,33).  visible(fra1gt_0,33). happensAt(walking(fra1gt_0),33). orientation(fra1gt_1,103,33). coords(fra1gt_1,279,10,33).  visible(fra1gt_1,33). happensAt(walking(fra1gt_1),33).
orientation(fra1gt_0,156,34). coords(fra1gt_0,210,169,34).  visible(fra1gt_0,34). happensAt(walking(fra1gt_0),34). orientation(fra1gt_1,103,34). coords(fra1gt_1,279,8,34).  visible(fra1gt_1,34). happensAt(walking(fra1gt_1),34).
orientation(fra1gt_0,156,35). coords(fra1gt_0,214,171,35).  visible(fra1gt_0,35). happensAt(walking(fra1gt_0),35). orientation(fra1gt_1,103,35). coords(fra1gt_1,279,6,35).  visible(fra1gt_1,35). happensAt(walking(fra1gt_1),35).
orientation(fra1gt_0,156,36). coords(fra1gt_0,217,174,36).  visible(fra1gt_0,36). happensAt(walking(fra1gt_0),36). orientation(fra1gt_1,103,36). coords(fra1gt_1,281,6,36).  visible(fra1gt_1,36). happensAt(walking(fra1gt_1),36).
#show holdsAt/2.